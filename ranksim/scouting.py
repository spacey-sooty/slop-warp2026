"""Pull pit-scouting data from the scoutinapp public export and reduce it to
per-team numbers the model can use.

The app (https://scoutinapp.vercel.app, source github.com/Hazzer890/scoutinapp)
serves a read-only dump of one event over HTTP:

    GET <deployment>.convex.site/api/scouting[?event=<tbaKey>]

It is pit scouting, not match scouting -- capability estimates filed once per
team per scout, not per-match performance. So it cannot replace the on-field
fit; what it can do is tell the fit what to believe about a team it has barely
seen, and give the climb model a signal it otherwise has almost no data for.

Reports are opinions, so every per-team figure here is a **median** across
scouts (one scout writing 200 balls/match should not move a team much) and every
use of them downstream is calibrated against on-field results rather than
trusted at face value.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .event import EventState

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://third-cow-432.convex.site/api/scouting"

# Picklists in the app are tiered, and an entry's `rank` is its position *within*
# its tier -- so a D-tier team and an S-tier team both hold rank 0. Averaging raw
# ranks across lists is meaningless; entries have to be flattened to an overall
# position first. Order from the app's own TIERS constant.
TIER_ORDER = ["S", "A", "B", "C", "D", "DNP"]


class ScoutingError(RuntimeError):
    pass


@dataclass
class TeamScouting:
    team: str
    reports: int = 0
    balls_per_match: float = 0.0
    auto_balls: float = 0.0
    storage: float = 0.0
    driver_rating: float = 0.0
    defense_rating: float = 0.0
    can_climb_rate: float = 0.0
    auto_climb_rate: float = 0.0
    can_score_rate: float = 0.0
    has_auto_rate: float = 0.0
    tags: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    picklist_rank: float | None = None  # mean overall position across personal lists
    picklist_lists: int = 0
    picklist_tier: str | None = None  # most common tier across personal lists
    primary_tier: str | None = None
    primary_rank: int | None = None  # overall position on the admin-owned list

    def as_dict(self) -> dict:
        return {
            "team": self.team,
            "reports": self.reports,
            "ballsPerMatch": self.balls_per_match,
            "autoBalls": self.auto_balls,
            "storage": self.storage,
            "driverRating": self.driver_rating,
            "defenseRating": self.defense_rating,
            "canClimbRate": self.can_climb_rate,
            "autoClimbRate": self.auto_climb_rate,
            "canScoreRate": self.can_score_rate,
            "hasAutoRate": self.has_auto_rate,
            "tags": self.tags,
            "notes": self.notes,
            "picklistRank": self.picklist_rank,
            "picklistLists": self.picklist_lists,
            "picklistTier": self.picklist_tier,
            "primaryTier": self.primary_tier,
            "primaryRank": self.primary_rank,
        }


@dataclass
class Scouting:
    event_key: str
    event_name: str
    teams: dict[str, TeamScouting]
    unmatched: list[str]
    total_reports: int
    picklists: int
    fetched_at: float

    def get(self, team: str) -> TeamScouting | None:
        return self.teams.get(team)

    def summary(self) -> dict:
        return {
            "eventKey": self.event_key,
            "eventName": self.event_name,
            "totalReports": self.total_reports,
            "picklists": self.picklists,
            "teamsCovered": sum(1 for t in self.teams.values() if t.reports),
            "unmatched": self.unmatched,
            "teams": {k: v.as_dict() for k, v in self.teams.items()},
        }


def fetch_raw(
    url: str = DEFAULT_URL,
    event_key: str | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    ttl: float = 120.0,
    refresh: bool = False,
) -> dict:
    cache_path = (cache_dir / "scouting.json") if cache_dir else None
    fresh = (
        cache_path is not None
        and cache_path.exists()
        and not refresh
        and (time.time() - cache_path.stat().st_mtime) < ttl
    )
    if offline or fresh:
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())
        if offline:
            raise ScoutingError(f"offline and no cached scouting dump at {cache_path}")

    target = f"{url}?event={event_key}" if event_key else url
    try:
        with urllib.request.urlopen(target, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text())
        raise ScoutingError(f"scouting fetch failed for {target}: {exc}") from exc

    if "error" in payload:
        raise ScoutingError(f"scouting export: {payload['error']}")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    return payload


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


def build(state: EventState, raw: dict, max_notes: int = 6) -> Scouting:
    """Reduce the raw dump to per-team figures, keyed by this event's team ids.

    The scouting app keys teams by plain number, so the same id quirk the CSV has
    shows up here: `9982` is the team TBA calls `4788B` (its nickname, "Can't
    Control Pro Max", to 4788's "Can't Control", gives the game away). The alias
    worked out while reconciling the CSV is reused rather than re-derived.
    """
    alias = dict(state.csv_aliases)
    known = set(state.teams)

    by_team: dict[str, list[dict]] = {}
    unmatched: Counter[str] = Counter()
    for report in raw.get("pitReports", []):
        number = report.get("teamNumber")
        if number is None:
            continue
        key = alias.get(str(number), str(number))
        if key not in known:
            unmatched[str(number)] += 1
            continue
        by_team.setdefault(key, []).append(report)

    # Picklists: consensus position across personal lists, plus the admin-owned
    # primary list's tier and position. Entries are flattened to an overall
    # 1-based position per list before anything is averaged (see TIER_ORDER).
    def ordered(entries: list[dict]) -> list[tuple[str, dict]]:
        rows = []
        for entry in entries:
            number = entry.get("teamNumber")
            if number is None:
                continue
            key = alias.get(str(number), str(number))
            if key in known:
                rows.append((key, entry))
        rows.sort(
            key=lambda kv: (
                TIER_ORDER.index(kv[1]["tier"])
                if kv[1]["tier"] in TIER_ORDER
                else len(TIER_ORDER),
                kv[1]["rank"],
            )
        )
        return rows

    personal_positions: dict[str, list[int]] = {}
    personal_tiers: dict[str, list[str]] = {}
    primary: dict[str, tuple[int, str]] = {}
    for plist in raw.get("picklists", []):
        rows = ordered(plist.get("entries", []))
        for position, (key, entry) in enumerate(rows, start=1):
            if plist.get("kind") == "primary":
                primary[key] = (position, entry["tier"])
            else:
                personal_positions.setdefault(key, []).append(position)
                personal_tiers.setdefault(key, []).append(entry["tier"])

    teams: dict[str, TeamScouting] = {}
    for team in state.teams:
        reports = by_team.get(team, [])
        scouting = TeamScouting(team=team, reports=len(reports))
        if reports:
            scouting.balls_per_match = _median(
                [r["ballsPerMatch"] for r in reports if r.get("ballsPerMatch") is not None]
            )
            scouting.auto_balls = _median(
                [r["autoBalls"] for r in reports if r.get("autoBalls") is not None]
            )
            scouting.storage = _median(
                [r["storageCapacity"] for r in reports if r.get("storageCapacity") is not None]
            )
            scouting.driver_rating = statistics.fmean(r["driverRating"] for r in reports)
            scouting.defense_rating = statistics.fmean(r["defenseRating"] for r in reports)
            scouting.can_climb_rate = _rate([bool(r.get("canClimb")) for r in reports])
            scouting.auto_climb_rate = _rate(
                [bool(r["autoClimb"]) for r in reports if r.get("autoClimb") is not None]
            )
            scouting.can_score_rate = _rate([bool(r.get("canScoreBalls")) for r in reports])
            scouting.has_auto_rate = _rate([bool(r.get("hasAuto")) for r in reports])
            scouting.tags = Counter(
                tag.strip().lower() for r in reports for tag in r.get("tags", [])
            ).most_common(6)
            scouting.notes = [
                r["notes"].strip()
                for r in reports
                if (r.get("notes") or "").strip()
            ][:max_notes]

        positions = personal_positions.get(team, [])
        if positions:
            scouting.picklist_rank = statistics.fmean(positions)
            scouting.picklist_lists = len(positions)
            tiers = personal_tiers.get(team, [])
            if tiers:
                scouting.picklist_tier = Counter(tiers).most_common(1)[0][0]
        if team in primary:
            scouting.primary_rank, scouting.primary_tier = primary[team]
        teams[team] = scouting

    return Scouting(
        event_key=raw.get("event", {}).get("tbaKey", ""),
        event_name=raw.get("event", {}).get("name", ""),
        teams=teams,
        unmatched=[f"{team} ({n} reports)" for team, n in sorted(unmatched.items())],
        total_reports=len(raw.get("pitReports", [])),
        picklists=len(raw.get("picklists", [])),
        fetched_at=raw.get("exportedAt", 0) / 1000.0,
    )


def load(
    state: EventState,
    url: str = DEFAULT_URL,
    cache_dir: Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> Scouting:
    raw = fetch_raw(
        url,
        event_key=state.event_key,
        cache_dir=cache_dir,
        offline=offline,
        refresh=refresh,
    )
    dump_key = raw.get("event", {}).get("tbaKey")
    if dump_key and dump_key != state.event_key:
        raise ScoutingError(
            f"scouting export is for {dump_key}, not {state.event_key}"
        )
    return build(state, raw)
