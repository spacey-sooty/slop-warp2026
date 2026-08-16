"""Event state: the qual schedule, the current standings, and the ranking rules.

Ranking rules for the 2026 game, reverse-engineered from TBA score breakdowns and
verified against six 2026 events (0 mismatches over 461 qual matches):

  Ranking points   3 win / 1 tie / 0 loss, plus one RP each for the
                   Energized, Supercharged and Traversal achievements.
  Match points     alliance totalPoints minus foulPoints (fouls the opponent
                   committed count toward winning the match but not toward the
                   ranking tiebreaker).
  Avg Auto Fuel    hubScore.autoPoints.
  Avg Tower        totalTowerPoints.
  Sort             Ranking Score (RP/played), then Avg Match, Avg Auto Fuel,
                   Avg Tower -- all averages over counting matches.

Surrogate appearances score for the alliance but do not count toward the
surrogate team's own ranking, and are excluded here accordingly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

# Team ids that differ between sources, seeded from data/aliases.json. Kept
# separate from the CSV reconciliation because it is a fact about team identity,
# not about the standings file -- CI builds without the CSV.
ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "aliases.json"


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    try:
        return dict(json.loads(Path(path).read_text()).get("aliases", {}))
    except (json.JSONDecodeError, AttributeError):
        return {}

WIN_RP = 3
TIE_RP = 1

# Per-robot tower values, read off the breakdowns (endgame Level1/2/3 = 10/20/30,
# auto Level1 = 15; auto Level2/3 never appeared in 2026 data, extrapolated).
AUTO_TOWER_VALUE = {"None": 0, "Level1": 15, "Level2": 30, "Level3": 45}
ENDGAME_TOWER_VALUE = {"None": 0, "Level1": 10, "Level2": 20, "Level3": 30}

SORT_KEYS = ("rank_score", "avg_match", "avg_auto", "avg_tower")


@dataclass
class TeamRecord:
    team: str
    rp: int = 0
    match_points: int = 0
    auto_fuel: int = 0
    tower: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    dq: int = 0
    played: int = 0

    def copy(self) -> "TeamRecord":
        return TeamRecord(**vars(self))

    @property
    def sort_orders(self) -> tuple[float, float, float, float]:
        n = self.played or 1
        return (self.rp / n, self.match_points / n, self.auto_fuel / n, self.tower / n)

    def as_dict(self) -> dict:
        rs, am, aa, at = self.sort_orders
        return {
            "team": self.team,
            "rp": self.rp,
            "matchPoints": self.match_points,
            "autoFuel": self.auto_fuel,
            "tower": self.tower,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "played": self.played,
            "rankScore": rs,
            "avgMatch": am,
            "avgAuto": aa,
            "avgTower": at,
        }


@dataclass
class Match:
    key: str
    number: int
    red: list[str]
    blue: list[str]
    red_surrogates: set[str] = field(default_factory=set)
    blue_surrogates: set[str] = field(default_factory=set)
    played: bool = False
    # Populated for played matches.
    winner: str = ""  # "red" | "blue" | "" (tie)
    breakdown: dict = field(default_factory=dict)

    def alliance(self, color: str) -> list[str]:
        return self.red if color == "red" else self.blue

    def counting(self, color: str) -> list[str]:
        sur = self.red_surrogates if color == "red" else self.blue_surrogates
        return [t for t in self.alliance(color) if t not in sur]

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "number": self.number,
            "red": self.red,
            "blue": self.blue,
            "redSurrogates": sorted(self.red_surrogates),
            "blueSurrogates": sorted(self.blue_surrogates),
            "played": self.played,
            "winner": self.winner,
        }


@dataclass
class AllianceResult:
    """One played alliance-appearance, the unit the model is fitted on."""

    match_key: str
    match_number: int
    color: str
    teams: list[str]
    counting_teams: list[str]
    hub_points: int
    hub_auto: int
    tower_points: int
    foul_points: int
    total_points: int
    match_points: int
    rp: int
    energized: bool
    supercharged: bool
    traversal: bool
    auto_tower_levels: list[str]
    endgame_tower_levels: list[str]
    won: bool
    tied: bool


def team_num(team_key: str) -> str:
    """'frc4788B' -> '4788B'."""
    return team_key[3:] if team_key.startswith("frc") else team_key


def parse_rankings(payload: dict | None) -> dict | None:
    """TBA's own published standings, reduced to what a cross-check needs.

    The standings here are rebuilt from the score breakdowns rather than read
    from this endpoint -- that is the only way to project them forward. But TBA
    computes the same thing from the same matches, so its published table is a
    free second opinion on the ranking rules, and worth carrying around to check
    against rather than trusting the rebuild blindly.
    """
    rows = (payload or {}).get("rankings") or []
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: r["rank"])
    teams = {}
    for row in rows:
        record = row.get("record") or {}
        extra = row.get("extra_stats") or []
        teams[team_num(row["team_key"])] = {
            "rank": row["rank"],
            "played": row.get("matches_played", 0),
            "rp": extra[0] if extra else None,
            "wins": record.get("wins", 0),
            "losses": record.get("losses", 0),
            "ties": record.get("ties", 0),
        }
    return {"order": [team_num(r["team_key"]) for r in ordered], "teams": teams}


@dataclass
class EventState:
    event_key: str
    event_name: str
    teams: list[str]
    matches: list[Match]
    results: list[AllianceResult]
    records: dict[str, TeamRecord]
    # TBA's published rankings, when the pull reached them: a reference to check
    # the rebuilt standings against, never the source of them.
    tba_rankings: dict | None = None
    csv_records: dict[str, TeamRecord] | None = None
    csv_aliases: dict[str, str] = field(default_factory=dict)
    csv_discrepancies: list[str] = field(default_factory=list)
    raw_matches: list[dict] = field(default_factory=list)
    # "tba" if this pull reached TBA, "cache"/"stale-cache" if it did not.
    tba_source: str = "cache"
    tba_warnings: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> list[Match]:
        return [m for m in self.matches if not m.played]

    @property
    def played(self) -> list[Match]:
        return [m for m in self.matches if m.played]


def _alliance_result(match: dict, color: str) -> AllianceResult:
    alliance = match["alliances"][color]
    breakdown = match["score_breakdown"][color]
    hub = breakdown["hubScore"]
    teams = [team_num(t) for t in alliance["team_keys"]]
    sur = {team_num(t) for t in alliance["surrogate_team_keys"]}
    return AllianceResult(
        match_key=match["key"],
        # Position in the schedule, which is what recency weighting decays over.
        match_number=match["match_number"],
        color=color,
        teams=teams,
        counting_teams=[t for t in teams if t not in sur],
        hub_points=hub["totalPoints"],
        hub_auto=hub["autoPoints"],
        tower_points=breakdown["totalTowerPoints"],
        foul_points=breakdown["foulPoints"],
        total_points=breakdown["totalPoints"],
        match_points=breakdown["totalPoints"] - breakdown["foulPoints"],
        rp=breakdown["rp"],
        energized=bool(breakdown["energizedAchieved"]),
        supercharged=bool(breakdown["superchargedAchieved"]),
        traversal=bool(breakdown["traversalAchieved"]),
        auto_tower_levels=[breakdown[f"autoTowerRobot{i}"] for i in (1, 2, 3)],
        endgame_tower_levels=[breakdown[f"endGameTowerRobot{i}"] for i in (1, 2, 3)],
        won=match["winning_alliance"] == color,
        tied=match["winning_alliance"] == "",
    )


def build_state(
    event_key: str, event_name: str, raw_matches: list[dict], csv_path: Path | None = None
) -> EventState:
    quals = [m for m in raw_matches if m["comp_level"] == "qm"]
    quals.sort(key=lambda m: m["match_number"])

    matches: list[Match] = []
    results: list[AllianceResult] = []
    records: dict[str, TeamRecord] = {}

    def record(team: str) -> TeamRecord:
        return records.setdefault(team, TeamRecord(team))

    for raw in quals:
        red = [team_num(t) for t in raw["alliances"]["red"]["team_keys"]]
        blue = [team_num(t) for t in raw["alliances"]["blue"]["team_keys"]]
        match = Match(
            key=raw["key"],
            number=raw["match_number"],
            red=red,
            blue=blue,
            red_surrogates={team_num(t) for t in raw["alliances"]["red"]["surrogate_team_keys"]},
            blue_surrogates={team_num(t) for t in raw["alliances"]["blue"]["surrogate_team_keys"]},
            played=bool(raw.get("score_breakdown")),
        )
        for team in red + blue:
            record(team)
        if match.played:
            match.winner = raw["winning_alliance"]
            for color in ("red", "blue"):
                res = _alliance_result(raw, color)
                results.append(res)
                dq = {team_num(t) for t in raw["alliances"][color]["dq_team_keys"]}
                for team in res.counting_teams:
                    r = record(team)
                    r.played += 1
                    r.rp += res.rp
                    r.match_points += res.match_points
                    r.auto_fuel += res.hub_auto
                    r.tower += res.tower_points
                    r.wins += res.won
                    r.ties += res.tied
                    r.losses += not (res.won or res.tied)
                    r.dq += team in dq
        matches.append(match)

    known = set(records)
    seeded = {
        source: target
        for source, target in load_aliases().items()
        if target in known and source not in known
    }

    state = EventState(
        event_key=event_key,
        event_name=event_name,
        teams=sorted(records, key=_team_sort_key),
        matches=matches,
        results=results,
        records=records,
        raw_matches=quals,
        csv_aliases=seeded,
    )
    if csv_path and Path(csv_path).exists():
        _attach_csv(state, Path(csv_path))
    return state


def _team_sort_key(team: str) -> tuple[int, str]:
    digits = "".join(c for c in team if c.isdigit())
    return (int(digits) if digits else 0, team)


def _attach_csv(state: EventState, csv_path: Path) -> None:
    """Load the hand-supplied standings CSV and reconcile its team ids with TBA's.

    The CSV that ships with this event lists a team as `9982` where TBA has
    `4788B`, so ids that appear on only one side are matched up by their ranking
    row (played/RP/W/L) instead of being dropped.
    """
    rows: dict[str, TeamRecord] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            team = row["TeamId"].strip()
            rows[team] = TeamRecord(
                team=team,
                rp=int(row["RankingPoints"]),
                match_points=int(row["MatchPoints"]),
                auto_fuel=int(row["AutoFuelPoints"]),
                tower=int(row["TowerPoints"]),
                wins=int(row["Wins"]),
                losses=int(row["Losses"]),
                ties=int(row["Ties"]),
                dq=int(row["Disqualifications"]),
                played=int(row["Played"]),
            )

    # Aliases seeded from data/aliases.json already account for both sides of the
    # pair, so neither end should be reported as missing.
    aliases: dict[str, str] = dict(state.csv_aliases)
    csv_only = [t for t in rows if t not in state.records and t not in aliases]
    tba_only = [
        t for t in state.records if t not in rows and t not in set(aliases.values())
    ]

    def signature(rec: TeamRecord) -> tuple:
        return (rec.played, rec.rp, rec.wins, rec.losses, rec.ties, rec.match_points)

    for csv_team in list(csv_only):
        sig = signature(rows[csv_team])
        hits = [t for t in tba_only if signature(state.records[t]) == sig]
        if len(hits) == 1:
            aliases[csv_team] = hits[0]
            tba_only.remove(hits[0])
            csv_only.remove(csv_team)

    canonical: dict[str, TeamRecord] = {}
    for csv_team, rec in rows.items():
        target = aliases.get(csv_team, csv_team)
        rec.team = target
        canonical[target] = rec

    discrepancies: list[str] = []
    for team in csv_only:
        discrepancies.append(f"{team}: in CSV but not in the TBA schedule")
    for team in tba_only:
        discrepancies.append(f"{team}: in the TBA schedule but not in the CSV")
    for team, rec in canonical.items():
        live = state.records.get(team)
        if live is None:
            continue
        for label, a, b in (
            ("played", rec.played, live.played),
            ("RP", rec.rp, live.rp),
            ("match points", rec.match_points, live.match_points),
            ("auto fuel", rec.auto_fuel, live.auto_fuel),
            ("tower", rec.tower, live.tower),
            ("record", (rec.wins, rec.losses, rec.ties), (live.wins, live.losses, live.ties)),
        ):
            if a != b:
                discrepancies.append(f"{team}: CSV {label} {a} vs TBA {b}")

    state.csv_records = canonical
    state.csv_aliases = aliases
    state.csv_discrepancies = discrepancies


def published_mismatches(
    records: dict[str, TeamRecord], published: dict | None
) -> list[str] | None:
    """Where the rebuilt standings disagree with TBA's published table.

    None if there is nothing to compare against. The mirror of
    verifyAgainstPublished in web/lib/standings.js, which does the same check in
    the browser after rebuilding the standings there.
    """
    if not published or not published.get("order"):
        return None
    mismatches = []
    order = rank_teams({t: records[t] for t in published["order"] if t in records})
    for i, team in enumerate(published["order"]):
        if i >= len(order) or order[i] != team:
            got = order[i] if i < len(order) else "nothing"
            mismatches.append(f"rank {i + 1}: TBA has {team}, rebuild has {got}")
    for team, row in (published.get("teams") or {}).items():
        record = records.get(team)
        if record is None:
            mismatches.append(f"{team}: ranked by TBA but not in the rebuild")
            continue
        for key, value in (
            ("played", row.get("played")),
            ("rp", row.get("rp")),
            ("wins", row.get("wins")),
            ("losses", row.get("losses")),
            ("ties", row.get("ties")),
        ):
            if value is not None and getattr(record, key) != value:
                mismatches.append(
                    f"{team} {key}: TBA {value}, rebuild {getattr(record, key)}"
                )
    return mismatches


def rank_teams(records: dict[str, TeamRecord]) -> list[str]:
    """Teams in official ranking order (best first)."""
    return sorted(
        records,
        key=lambda t: (
            -records[t].sort_orders[0],
            -records[t].sort_orders[1],
            -records[t].sort_orders[2],
            -records[t].sort_orders[3],
            _team_sort_key(t),
        ),
    )
