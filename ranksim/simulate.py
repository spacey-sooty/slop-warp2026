"""Monte Carlo over the remaining qual matches.

One trial plays out every unplayed match, adds the result to the current
standings, re-sorts by the official tiebreaker chain, and records where each team
landed. Aggregating a few thousand of those gives the rank distribution.

Per trial the model draws:
  * a team rating vector (optionally resampled from each rating's standard error,
    so a team with two matches carries its uncertainty into the projection),
  * an alliance hub score around that rating, and its auto-fuel share,
  * a climb outcome per robot,
  * opponent-foul points, which decide close matches but never the tiebreakers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .event import TIE_RP, WIN_RP, EventState, Match, TeamRecord, rank_teams
from .model import Fit

# How many resamples to spend trying to hit a forced winner before nudging the
# score. 60 tries covers anything down to a ~5% upset without distortion.
FORCE_MAX_TRIES = 60

# Alliance scores are drawn from a normal truncated at zero: the weakest
# alliances here have a predicted hub score below one residual sigma, so an
# unclamped normal puts real mass on negative scores, and clamping instead of
# truncating piles an atom onto exactly 0 -- which showed up as a spurious 1%+
# rate of 0-0 ties. Scores are then rounded, since real scores are integers and
# that is what makes a genuine tie possible at all.
TRUNCATE_MAX_TRIES = 16


@dataclass
class SimOptions:
    n: int = 5000
    seed: int | None = None
    opr_uncertainty: bool = True
    cutoff: int = 8
    forced: dict[str, str] = field(default_factory=dict)
    source: str = "csv"  # "csv" | "tba" -- which current standings to start from
    use_scouting: bool = True  # recorded for the caller; the fit already reflects it


def _base_records(state: EventState, source: str) -> dict[str, TeamRecord]:
    if source == "csv" and state.csv_records:
        base = {t: state.csv_records[t].copy() for t in state.csv_records}
        for team in state.teams:
            base.setdefault(team, TeamRecord(team))
        return base
    return {t: state.records[t].copy() for t in state.teams}


def _percentile_from_counts(counts: list[int], total: int, q: float) -> int:
    target = q * total
    seen = 0
    for rank, c in enumerate(counts, start=1):
        seen += c
        if seen >= target:
            return rank
    return len(counts)


def simulate(state: EventState, fit: Fit, options: SimOptions) -> dict:
    rng = random.Random(options.seed)
    gauss = rng.gauss

    teams = list(state.teams)
    n_teams = len(teams)
    index = {t: i for i, t in enumerate(teams)}

    base = _base_records(state, options.source)
    base_rp = [base[t].rp for t in teams]
    base_mp = [base[t].match_points for t in teams]
    base_af = [base[t].auto_fuel for t in teams]
    base_tw = [base[t].tower for t in teams]
    base_played = [base[t].played for t in teams]

    hub_rating = [fit.hub[t] for t in teams]
    hub_se = [fit.hub_se[t] for t in teams]
    auto_rating = [fit.auto[t] for t in teams]
    auto_se = [fit.auto_se[t] for t in teams]

    # Per-team climb sampler: cumulative weights over (points) outcomes.
    tower_points: list[list[int]] = []
    tower_cum: list[list[float]] = []
    for t in teams:
        outcomes = fit.tower_outcomes[t]
        pts, cum, running = [], [], 0.0
        for outcome, prob in outcomes:
            running += prob
            pts.append(outcome.points)
            cum.append(running)
        if cum:
            cum[-1] = 1.0
        else:
            pts, cum = [0], [1.0]
        tower_points.append(pts)
        tower_cum.append(cum)

    fouls = fit.foul_samples or [0]
    thr_energized = fit.thresholds["energized"]
    thr_supercharged = fit.thresholds["supercharged"]
    thr_traversal = fit.thresholds["traversal"]
    sigma_hub = fit.sigma_hub
    sigma_auto = fit.sigma_auto

    remaining: list[Match] = state.remaining
    # Precompute index lists so the hot loop touches no strings.
    sched = []
    for m in remaining:
        sched.append(
            (
                m.key,
                [index[t] for t in m.red],
                [index[t] for t in m.blue],
                [index[t] for t in m.counting("red")],
                [index[t] for t in m.counting("blue")],
                options.forced.get(m.key, ""),
            )
        )

    rank_counts = [[0] * n_teams for _ in range(n_teams)]
    rank_sum = [0.0] * n_teams
    rp_sum = [0.0] * n_teams
    rs_sum = [0.0] * n_teams
    match_counts = {m.key: {"red": 0, "blue": 0, "tie": 0} for m in remaining}
    forced_hits = 0
    forced_nudges = 0

    def draw_score(mean: float, sigma: float) -> int:
        """Integer draw from Normal(mean, sigma) truncated at zero."""
        for _ in range(TRUNCATE_MAX_TRIES):
            value = mean + gauss(0.0, sigma)
            if value >= 0.0:
                return int(value + 0.5)
        return 0

    def draw_tower(idx: int) -> int:
        r = rng.random()
        cum = tower_cum[idx]
        for i, c in enumerate(cum):
            if r <= c:
                return tower_points[idx][i]
        return tower_points[idx][-1]

    for _ in range(options.n):
        if options.opr_uncertainty:
            hub_r = [hub_rating[i] + gauss(0.0, hub_se[i]) for i in range(n_teams)]
            auto_r = [auto_rating[i] + gauss(0.0, auto_se[i]) for i in range(n_teams)]
        else:
            hub_r = hub_rating
            auto_r = auto_rating

        rp = base_rp[:]
        mp = base_mp[:]
        af = base_af[:]
        tw = base_tw[:]
        played = base_played[:]

        for key, red, blue, red_count, blue_count, force in sched:
            for _try in range(FORCE_MAX_TRIES if force in ("red", "blue") else 1):
                r_hub = draw_score(sum(hub_r[i] for i in red), sigma_hub)
                b_hub = draw_score(sum(hub_r[i] for i in blue), sigma_hub)
                r_auto = draw_score(sum(auto_r[i] for i in red), sigma_auto)
                b_auto = draw_score(sum(auto_r[i] for i in blue), sigma_auto)
                if r_auto > r_hub:
                    r_auto = r_hub
                if b_auto > b_hub:
                    b_auto = b_hub
                r_tower = draw_tower(red[0]) + draw_tower(red[1]) + draw_tower(red[2])
                b_tower = draw_tower(blue[0]) + draw_tower(blue[1]) + draw_tower(blue[2])
                r_total = r_hub + r_tower + rng.choice(fouls)
                b_total = b_hub + b_tower + rng.choice(fouls)
                if not force or force == "tie":
                    break
                if (force == "red" and r_total > b_total) or (
                    force == "blue" and b_total > r_total
                ):
                    forced_hits += 1
                    break
            else:
                # Never landed the requested upset; push the forced winner over
                # the line by the smallest amount that flips it.
                forced_nudges += 1
                if force == "red":
                    bump = b_total - r_total + 1
                    r_hub += bump
                    r_total += bump
                else:
                    bump = r_total - b_total + 1
                    b_hub += bump
                    b_total += bump

            if force == "tie":
                red_win = blue_win = False
                tie = True
            else:
                red_win = r_total > b_total
                blue_win = b_total > r_total
                tie = not (red_win or blue_win)

            counts = match_counts[key]
            counts["red" if red_win else "blue" if blue_win else "tie"] += 1

            r_rp = (WIN_RP if red_win else TIE_RP if tie else 0)
            b_rp = (WIN_RP if blue_win else TIE_RP if tie else 0)
            if r_hub >= thr_energized:
                r_rp += 1
                if r_hub >= thr_supercharged:
                    r_rp += 1
            if b_hub >= thr_energized:
                b_rp += 1
                if b_hub >= thr_supercharged:
                    b_rp += 1
            if r_tower >= thr_traversal:
                r_rp += 1
            if b_tower >= thr_traversal:
                b_rp += 1

            r_mp = r_hub + r_tower
            b_mp = b_hub + b_tower
            r_af = r_auto
            b_af = b_auto
            for i in red_count:
                rp[i] += r_rp
                mp[i] += r_mp
                af[i] += r_af
                tw[i] += r_tower
                played[i] += 1
            for i in blue_count:
                rp[i] += b_rp
                mp[i] += b_mp
                af[i] += b_af
                tw[i] += b_tower
                played[i] += 1

        order = sorted(
            range(n_teams),
            key=lambda i: (
                -rp[i] / played[i] if played[i] else 0.0,
                -mp[i] / played[i] if played[i] else 0.0,
                -af[i] / played[i] if played[i] else 0.0,
                -tw[i] / played[i] if played[i] else 0.0,
                i,
            ),
        )
        for rank0, i in enumerate(order):
            rank_counts[i][rank0] += 1
            rank_sum[i] += rank0 + 1
            rp_sum[i] += rp[i]
            rs_sum[i] += rp[i] / played[i] if played[i] else 0.0

    n = options.n
    cutoff = max(1, min(options.cutoff, n_teams))
    current_order = {t: i + 1 for i, t in enumerate(rank_teams(base))}

    team_rows = []
    for i, team in enumerate(teams):
        counts = rank_counts[i]
        dist = [c / n for c in counts]
        team_rows.append(
            {
                "team": team,
                "currentRank": current_order.get(team, n_teams),
                "current": base[team].as_dict(),
                "meanRank": rank_sum[i] / n,
                "medianRank": _percentile_from_counts(counts, n, 0.5),
                "p05Rank": _percentile_from_counts(counts, n, 0.05),
                "p95Rank": _percentile_from_counts(counts, n, 0.95),
                "bestRank": next(r + 1 for r, c in enumerate(counts) if c),
                "worstRank": next(
                    r + 1 for r in range(n_teams - 1, -1, -1) if counts[r]
                ),
                "pRank1": dist[0],
                "pTop3": sum(dist[:3]),
                "pCutoff": sum(dist[:cutoff]),
                "expectedRp": rp_sum[i] / n,
                "expectedRankScore": rs_sum[i] / n,
                "rankDist": dist,
                "remainingMatches": [
                    m.key for m in remaining if team in m.counting("red") + m.counting("blue")
                ],
            }
        )
    team_rows.sort(key=lambda r: (r["meanRank"], -r["pCutoff"]))

    match_rows = []
    for m in remaining:
        c = match_counts[m.key]
        match_rows.append(
            {
                **m.as_dict(),
                "pRed": c["red"] / n,
                "pBlue": c["blue"] / n,
                "pTie": c["tie"] / n,
                "forced": options.forced.get(m.key, ""),
                "redPredicted": sum(fit.hub[t] for t in m.red),
                "bluePredicted": sum(fit.hub[t] for t in m.blue),
            }
        )

    return {
        "teams": team_rows,
        "matches": match_rows,
        "meta": {
            "n": n,
            "seed": options.seed,
            "cutoff": cutoff,
            "oprUncertainty": options.opr_uncertainty,
            "source": options.source,
            "useScouting": options.use_scouting,
            "forced": options.forced,
            "forcedResampled": forced_hits,
            "forcedNudged": forced_nudges,
            "remainingMatches": len(remaining),
        },
    }
