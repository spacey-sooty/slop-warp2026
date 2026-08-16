"""Fit a predictive model of alliance performance from the played qual matches.

Everything the simulator needs is estimated here:

  hub / auto OPR   ridge-regularised least squares on alliance hub points and hub
                   auto points. Ridge shrinks toward the event mean, which matters
                   at 27 teams / 70 alliance-appearances where plain OPR is noisy.
  recency weights  every observation is discounted by how long ago it happened, so
                   a robot that fixed its intake after qm10 is rated on the matches
                   since rather than on its whole event.
  residual sigma   spread of a single alliance's score around its predicted value.
  parameter SE     how well each team's rating is pinned down; optionally resampled
                   per simulation so a team with two matches is not treated as a
                   known quantity.
  tower model      per-robot empirical distribution over climb outcomes, smoothed
                   toward the event-wide pool.
  foul model       empirical draw of opponent-foul points, which swing who wins a
                   match but never the ranking tiebreakers.
  RP thresholds    learned from this event's own achieved/not-achieved boundary.

Recency weighting applies to everything that estimates *a team*: the two OPR
fits, their residual sigma and standard errors, the climb distribution and the
scouted climb calibration. It deliberately does not touch the RP thresholds or
the foul draw -- a threshold is a fixed rule of the game being read off the
data, not a quantity that drifts, and fouls are a property of the opponent.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .event import (
    AUTO_TOWER_VALUE,
    ENDGAME_TOWER_VALUE,
    AllianceResult,
    EventState,
)
from .linalg import chol_inv_diag, chol_solve, cholesky, normal_equations

# Fallbacks when this event has never shown the achievement. The 2026 regional
# values (energized 100 hub, supercharged 360 hub, traversal 50 tower) hold across
# every regional checked; this offseason event runs a shortened match and sits at
# 75 / ~213, so a fallback is only a last resort.
DEFAULT_THRESHOLDS = {"energized": 100.0, "supercharged": 360.0, "traversal": 50.0}

# Weight of the event-wide pooled tower distribution in each team's own estimate,
# in units of "extra matches". Climbs are rare, so a team with one climb in eight
# matches should not be modelled as a 12.5% climber on that evidence alone.
TOWER_PRIOR_WEIGHT = 4.0

# How much of a scouted climb capability to believe as a per-match rate before
# the event has produced any climbs to calibrate against.
DEFAULT_CLIMB_TRUST = 0.5

# Floor on the scouted climb prior. A pit report is a snapshot of one afternoon;
# robots get hangers bolted on mid-event. Without this, a team scouted as
# non-climbing who has not yet climbed gets probability exactly zero, and the
# simulation treats their climbing as impossible rather than unlikely.
MIN_CLIMB_PRIOR = 0.02

# Recency: how many qualification matches it takes for an observation to count
# half as much as one played now. Robots at an event are not a fixed quantity --
# they break, get repaired, and their drivers get better -- so the last few
# matches say more about the rest of the event than the first few do.
#
# Picked by walk-forward backtest, and deliberately not by how much it helps
# here alone. 2026auwarp's own 20 out-of-sample matches prefer a much shorter
# memory (half-life 6-12), but replaying the same test over the three reference
# regionals shows that setting is actively harmful on a 66-80 match schedule,
# where each team's matches are spread thin enough that a short half-life throws
# away most of their record. 20 is the longest setting that still improves
# 2026auwarp (Brier 0.157 -> 0.150) while leaving the regionals where they were.
# Longer schedules want a longer half-life; 0 turns the decay off entirely.
DEFAULT_HALF_LIFE = 20.0


@dataclass
class TowerOutcome:
    auto_level: str
    endgame_level: str
    points: int


def recency_weights(
    results: list[AllianceResult], half_life: float = DEFAULT_HALF_LIFE
) -> list[float]:
    """Per-observation weight, halving every `half_life` matches into the past.

    Age is counted in schedule position, so both alliances of the same match get
    the same weight and a team's own match count never enters into it. The
    weights are then rescaled to average 1, which is what keeps the rest of the
    model on its usual scale: ridge strength, residual sigma and the standard
    errors all read against the number of observations, and a decay that shrank
    the total weight would quietly turn into extra shrinkage.

    Only the *gap* between matches matters, not what the reference point is --
    rescaling cancels any common factor -- so this is the same whether it is
    called mid-event or at the end.
    """
    if not results or half_life is None or half_life <= 0:
        return [1.0] * len(results)
    latest = max(r.match_number for r in results)
    raw = [0.5 ** ((latest - r.match_number) / half_life) for r in results]
    total = sum(raw)
    if total <= 0:
        return [1.0] * len(results)
    scale = len(raw) / total
    return [w * scale for w in raw]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return statistics.fmean(values) if values else 0.0
    return sum(w * v for w, v in zip(weights, values)) / total


@dataclass
class Fit:
    teams: list[str]
    hub: dict[str, float]
    hub_se: dict[str, float]
    auto: dict[str, float]
    auto_se: dict[str, float]
    sigma_hub: float
    sigma_auto: float
    tower_outcomes: dict[str, list[tuple[TowerOutcome, float]]]
    foul_samples: list[int]
    thresholds: dict[str, float]
    threshold_sources: dict[str, str]
    ridge: float
    n_observations: int
    mean_hub: float
    r_squared: float
    scouting: dict = field(default_factory=lambda: {"used": False})
    recency: dict = field(default_factory=lambda: {"applied": False, "halfLife": 0.0})

    def summary(self) -> dict:
        return {
            "sigmaHub": self.sigma_hub,
            "sigmaAuto": self.sigma_auto,
            "ridge": self.ridge,
            "observations": self.n_observations,
            "meanAllianceHub": self.mean_hub,
            "rSquared": self.r_squared,
            "scouting": self.scouting,
            "recency": self.recency,
            "thresholds": self.thresholds,
            "thresholdSources": self.threshold_sources,
            "teams": {
                t: {
                    "hubOpr": self.hub[t],
                    "hubSe": self.hub_se[t],
                    "autoOpr": self.auto[t],
                    "towerRate": self.tower_rate(t),
                    "expectedTower": self.expected_tower(t),
                }
                for t in self.teams
            },
        }

    def tower_rate(self, team: str) -> float:
        return sum(p for o, p in self.tower_outcomes[team] if o.points > 0)

    def expected_tower(self, team: str) -> float:
        return sum(o.points * p for o, p in self.tower_outcomes[team])


def _fit_component(
    results: list[AllianceResult],
    teams: list[str],
    index: dict[str, int],
    values: list[float],
    ridge: float,
    prior: dict[str, float] | None = None,
    weights: list[float] | None = None,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Weighted ridge least squares on alliance totals.

    Minimises sum_i w_i (x_i b - y_i)^2 + ridge*||b - p||^2, where p is the
    shrinkage target -- the flat event mean by default, or a per-team prior (from
    scouting) when one is supplied -- and w is the recency weight. Shrinking
    toward "what scouts expect of this team" rather than "what the average team
    does" is what makes a two-match team's rating sane; weighting by w is what
    makes a team's rating follow the robot it is running now.

    The weights average 1, so with the decay off every w_i is exactly 1 and this
    is the plain unweighted fit, residual sigma and standard errors included.
    """
    rows = [[index[t] for t in r.teams] for r in results]
    w = [1.0] * len(values) if weights is None else weights
    mean_alliance = _weighted_mean(values, w)
    per_team_mean = mean_alliance / 3.0
    targets = {t: (prior.get(t, per_team_mean) if prior else per_team_mean) for t in teams}
    centered = [v - sum(targets[t] for t in r.teams) for v, r in zip(values, results)]

    ata, aty = normal_equations(rows, centered, len(teams), ridge, w)
    lower = cholesky(ata)
    deviations = chol_solve(lower, aty)
    ratings = {t: targets[t] + deviations[index[t]] for t in teams}

    residuals = [
        v - sum(ratings[t] for t in r.teams) for v, r in zip(values, results)
    ]
    dof = max(len(values) - len(teams), 1)
    sigma = math.sqrt(sum(wi * e * e for wi, e in zip(w, residuals)) / dof)

    total_ss = sum(wi * (v - mean_alliance) ** 2 for wi, v in zip(w, values))
    resid_ss = sum(wi * e * e for wi, e in zip(w, residuals))
    r_squared = 1.0 - resid_ss / total_ss if total_ss > 0 else 0.0

    inv_diag = chol_inv_diag(lower)
    se = {t: sigma * math.sqrt(max(inv_diag[index[t]], 0.0)) for t in teams}
    return ratings, se, sigma, r_squared


def _calibrate(x: dict[str, float], y: dict[str, float]) -> tuple[float, float, float]:
    """Least-squares line y = a + b*x over their shared keys, plus its r^2.

    Used to turn a scouted quantity into the units the model works in. The r^2 is
    the point of it: it becomes the weight the prior carries, so scouting that
    does not predict this event's results quietly stops influencing the fit.
    """
    keys = [k for k in x if k in y]
    if len(keys) < 3:
        return 0.0, 0.0, 0.0
    xs = [x[k] for k in keys]
    ys = [y[k] for k in keys]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(xs, ys))
    total = sum((b - my) ** 2 for b in ys)
    r_squared = 1.0 - resid / total if total > 0 else 0.0
    return intercept, slope, max(0.0, min(1.0, r_squared))


def _tower_model(
    results: list[AllianceResult],
    teams: list[str],
    climb_prior: dict[str, float] | None = None,
    weights: list[float] | None = None,
) -> dict[str, list[tuple[TowerOutcome, float]]]:
    """Per-robot distribution over climb outcomes.

    Climbs are the thinnest data at any event -- here, six in seventy alliance
    appearances -- so each team's own record is smoothed toward a prior. Without
    scouting that prior is the event-wide pool, which gives every team the same
    small climb chance whether or not their robot has a hanger. With scouting,
    the prior becomes the team's own scouted climb capability, so a robot nobody
    has seen climb but everyone reports *can* climb is modelled as a climber, and
    a robot with no hanger drops to ~zero instead of the pooled average.

    Observations carry their recency weight, which matters more here than
    anywhere else in the fit: a hanger bolted on at lunchtime shows up as climbs
    that all sit at the recent end, and an unweighted count buries them under a
    morning's worth of no-climbs.
    """
    w = [1.0] * len(results) if weights is None else weights
    per_team: dict[str, list[tuple[TowerOutcome, float]]] = {t: [] for t in teams}
    pooled: list[tuple[TowerOutcome, float]] = []
    for res, weight in zip(results, w):
        for slot, team in enumerate(res.teams):
            auto_level = res.auto_tower_levels[slot]
            end_level = res.endgame_tower_levels[slot]
            outcome = TowerOutcome(
                auto_level,
                end_level,
                AUTO_TOWER_VALUE.get(auto_level, 0) + ENDGAME_TOWER_VALUE.get(end_level, 0),
            )
            per_team.setdefault(team, []).append((outcome, weight))
            pooled.append((outcome, weight))

    no_climb = TowerOutcome("None", "None", 0)
    climbs = [(o, weight) for o, weight in pooled if o.points > 0]
    # If nobody at the event has climbed yet, a scouted climber is assumed to
    # manage the cheapest climb the game offers.
    climb_shapes = climbs or [
        (TowerOutcome("None", "Level1", ENDGAME_TOWER_VALUE["Level1"]), 1.0)
    ]

    def spread(pool: list[tuple[TowerOutcome, float]], mass: float):
        """Split `mass` over a pool of outcomes in proportion to their weights."""
        total = sum(weight for _, weight in pool)
        if total <= 0:
            return []
        return [(o, mass * weight / total) for o, weight in pool]

    def distribution(
        observations: list[tuple[TowerOutcome, float]],
        prior: list[tuple[TowerOutcome, float]],
    ):
        counts: dict[tuple[str, str], list] = {}
        for o, weight in list(observations) + list(prior):
            counts.setdefault((o.auto_level, o.endgame_level), [o, 0.0])[1] += weight
        total = sum(c for _, c in counts.values()) or 1.0
        return [(o, c / total) for o, c in counts.values()]

    def prior_for(team: str) -> list[tuple[TowerOutcome, float]]:
        if climb_prior is not None and team in climb_prior:
            p = max(0.0, min(1.0, climb_prior[team]))
            return spread(climb_shapes, TOWER_PRIOR_WEIGHT * p) + [
                (no_climb, TOWER_PRIOR_WEIGHT * (1.0 - p))
            ]
        return spread(pooled, TOWER_PRIOR_WEIGHT)

    return {t: distribution(per_team.get(t, []), prior_for(t)) for t in teams}


def _learn_threshold(
    achieved: list[float], missed: list[float], name: str
) -> tuple[float, str]:
    """Midpoint of the observed achieved/not-achieved boundary.

    With a hard in-game threshold the two sets separate cleanly, so the midpoint
    of (max missed, min achieved] is the tightest unbiased estimate available.
    """
    if achieved and missed:
        lo, hi = max(missed), min(achieved)
        if hi > lo:
            return (lo + hi) / 2.0, f"learned from {len(achieved)} achieved / {len(missed)} not"
        # Overlap means it is not a clean threshold on this quantity; fall back.
        return DEFAULT_THRESHOLDS[name], "overlapping observations, using 2026 default"
    if achieved:
        return min(achieved), f"lower bound from {len(achieved)} achieved"
    return DEFAULT_THRESHOLDS[name], "never achieved here, using 2026 default"


def _scouted_prior(
    scouted: dict[str, float],
    baseline: dict[str, float],
    teams: list[str],
    weight: float,
    label: str,
) -> tuple[dict[str, float] | None, dict]:
    """Turn a scouted quantity into a shrinkage target in the model's own units.

    The scouted number (say, median balls per match) is regressed onto the
    baseline on-field ratings to learn the conversion, and the fit's r^2 becomes
    how far each team is pulled toward it. That keeps the loop honest in both
    directions: scouting that tracks results carries weight, scouting that does
    not fades to the plain event mean, and the *scale* always comes from the
    field even though the *shape* comes from the pit.
    """
    intercept, slope, r_squared = _calibrate(scouted, baseline)
    mean = statistics.fmean(baseline.values())
    trust = max(0.0, min(1.0, weight * r_squared))
    info = {
        "metric": label,
        "intercept": intercept,
        "slope": slope,
        "rSquared": r_squared,
        "trust": trust,
        "teamsUsed": sum(1 for t in scouted if t in baseline),
    }
    if trust <= 0.0 or slope <= 0.0:
        info["applied"] = False
        return None, info
    prior = {}
    for team in teams:
        if team in scouted:
            predicted = max(0.0, intercept + slope * scouted[team])
            prior[team] = trust * predicted + (1.0 - trust) * mean
        else:
            prior[team] = mean
    info["applied"] = True
    return prior, info


def _climb_prior(
    results: list[AllianceResult],
    scouted: dict,
    weight: float,
    weights: list[float] | None = None,
) -> tuple[dict[str, float] | None, dict]:
    """Scouted climb capability -> a per-match climb probability.

    Capability is not frequency: "this robot can climb" does not mean it climbs
    every match. When the event has seen at least one climb, the conversion is
    calibrated against the observed per-robot climb rates. Until then there is
    nothing to calibrate against, so scouted capability is discounted by a flat
    factor rather than taken at face value.
    """
    w = [1.0] * len(results) if weights is None else weights
    observed: dict[str, list[tuple[bool, float]]] = {}
    for res, row_weight in zip(results, w):
        for slot, team in enumerate(res.teams):
            climbed = (
                res.auto_tower_levels[slot] != "None"
                or res.endgame_tower_levels[slot] != "None"
            )
            observed.setdefault(team, []).append((climbed, row_weight))
    rates = {
        t: sum(rw for c, rw in v if c) / sum(rw for _, rw in v)
        for t, v in observed.items()
        if v and sum(rw for _, rw in v) > 0
    }
    scouted_rates = {t: s.can_climb_rate for t, s in scouted.items()}

    info: dict = {"metric": "canClimb report rate", "teamsUsed": len(scouted_rates)}
    if any(rates.values()):
        intercept, slope, r_squared = _calibrate(scouted_rates, rates)
        info.update({"intercept": intercept, "slope": slope, "rSquared": r_squared})
        if slope > 0:
            info["mode"] = "calibrated against observed climbs"
            prior = {
                t: max(MIN_CLIMB_PRIOR, min(0.95, intercept + slope * r))
                for t, r in scouted_rates.items()
            }
            return prior, info
    info["mode"] = f"no climbs observed yet, discounted by {DEFAULT_CLIMB_TRUST}"
    prior = {
        t: max(MIN_CLIMB_PRIOR, min(0.95, DEFAULT_CLIMB_TRUST * weight * r))
        for t, r in scouted_rates.items()
    }
    return prior, info


def fit(
    state: EventState,
    ridge: float = 3.0,
    scouting=None,
    scouting_weight: float = 1.0,
    half_life: float = DEFAULT_HALF_LIFE,
) -> Fit:
    results = state.results
    teams = list(state.teams)
    index = {t: i for i, t in enumerate(teams)}
    if not results:
        raise ValueError("no played qual matches to fit on")

    weights = recency_weights(results, half_life)
    hub_values = [float(r.hub_points) for r in results]
    auto_values = [float(r.hub_auto) for r in results]

    hub, hub_se, sigma_hub, r_squared = _fit_component(
        results, teams, index, hub_values, ridge, weights=weights
    )
    auto, auto_se, sigma_auto, _ = _fit_component(
        results, teams, index, auto_values, ridge, weights=weights
    )

    scouting_info: dict = {"used": False}
    climb_prior: dict[str, float] | None = None
    if scouting is not None and scouting_weight > 0:
        scouted = {t: s for t, s in scouting.teams.items() if s.reports}
        hub_prior, hub_info = _scouted_prior(
            {t: s.balls_per_match for t, s in scouted.items()},
            hub,
            teams,
            scouting_weight,
            "median balls per match",
        )
        auto_prior, auto_info = _scouted_prior(
            {t: s.auto_balls for t, s in scouted.items()},
            auto,
            teams,
            scouting_weight,
            "median auto balls",
        )
        if hub_prior:
            hub, hub_se, sigma_hub, r_squared = _fit_component(
                results, teams, index, hub_values, ridge, prior=hub_prior, weights=weights
            )
        if auto_prior:
            auto, auto_se, sigma_auto, _ = _fit_component(
                results, teams, index, auto_values, ridge, prior=auto_prior, weights=weights
            )
        climb_prior, climb_info = _climb_prior(
            results, scouted, scouting_weight, weights=weights
        )
        scouting_info = {
            "used": True,
            "reports": scouting.total_reports,
            "teamsCovered": len(scouted),
            "weight": scouting_weight,
            "hub": hub_info,
            "auto": auto_info,
            "climb": climb_info,
        }

    thresholds: dict[str, float] = {}
    sources: dict[str, str] = {}
    for name, flag, value in (
        ("energized", "energized", lambda r: float(r.hub_points)),
        ("supercharged", "supercharged", lambda r: float(r.hub_points)),
        ("traversal", "traversal", lambda r: float(r.tower_points)),
    ):
        achieved = [value(r) for r in results if getattr(r, flag)]
        missed = [value(r) for r in results if not getattr(r, flag)]
        thresholds[name], sources[name] = _learn_threshold(achieved, missed, name)

    # Effective sample size: how many equally-weighted matches the discounted
    # ones are worth. The honest headline for "fitted on N appearances" once the
    # oldest of them count for a fraction of a match.
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    recency = {
        "applied": bool(half_life and half_life > 0),
        "halfLife": float(half_life or 0.0),
        "effectiveObservations": (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0,
        "oldestWeight": min(weights),
        "newestWeight": max(weights),
    }

    return Fit(
        teams=teams,
        hub=hub,
        hub_se=hub_se,
        auto=auto,
        auto_se=auto_se,
        sigma_hub=sigma_hub,
        sigma_auto=sigma_auto,
        tower_outcomes=_tower_model(results, teams, climb_prior, weights),
        foul_samples=[r.foul_points for r in results],
        thresholds=thresholds,
        threshold_sources=sources,
        ridge=ridge,
        n_observations=len(results),
        mean_hub=_weighted_mean([float(r.hub_points) for r in results], weights),
        r_squared=r_squared,
        scouting=scouting_info,
        recency=recency,
    )
