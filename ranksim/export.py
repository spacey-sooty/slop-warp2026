"""Bake the event into a JSON bundle the browser can fit and simulate from.

The static build has no server, so everything that needs a TBA key has to be
resolved ahead of time and shipped as data. That is exactly the played-match
record and the remaining schedule -- both of which only change when a match
finishes, i.e. at rebuild time.

Scouting deliberately does *not* go in here. Its export is public and serves
open CORS, so the page fetches it live from the deployed app and refits in the
browser; baking it would make the refresh button a lie.

What ships is the raw material for the fit, not the fit itself, so the browser
can refit against freshly pulled scouting rather than being stuck with whatever
priors were in force at build time.
"""

from __future__ import annotations

import json
from pathlib import Path

from .event import (
    AUTO_TOWER_VALUE,
    ENDGAME_TOWER_VALUE,
    TIE_RP,
    WIN_RP,
    EventState,
    rank_teams,
)
from .model import (
    DEFAULT_CLIMB_TRUST,
    DEFAULT_THRESHOLDS,
    MIN_CLIMB_PRIOR,
    TOWER_PRIOR_WEIGHT,
)
from .scouting import DEFAULT_URL as SCOUTING_URL
from .scouting import TIER_ORDER


def bundle(
    state: EventState,
    ridge: float = 3.0,
    scouting_url: str = SCOUTING_URL,
    generated_at: float = 0.0,
) -> dict:
    base = state.csv_records if state.csv_records else state.records
    order = rank_teams(base)

    return {
        "event": {
            "key": state.event_key,
            "name": state.event_name,
            "matchesPlayed": len(state.played),
            "matchesRemaining": len(state.remaining),
        },
        "generatedAt": generated_at,
        "tbaSource": state.tba_source,
        "tbaWarnings": state.tba_warnings,
        "scoutingUrl": scouting_url,
        "teams": list(state.teams),
        "standingsSource": "csv" if state.csv_records else "tba",
        "standings": {t: base[t].as_dict() for t in state.teams},
        "currentOrder": order,
        "csvAliases": state.csv_aliases,
        "csvDiscrepancies": state.csv_discrepancies,
        # One row per played alliance-appearance: the design matrix, the fit
        # targets, the climb record and the RP achievements, all in one.
        "results": [
            {
                "matchKey": r.match_key,
                "color": r.color,
                "teams": r.teams,
                "hubPoints": r.hub_points,
                "hubAuto": r.hub_auto,
                "towerPoints": r.tower_points,
                "foulPoints": r.foul_points,
                "matchPoints": r.match_points,
                "rp": r.rp,
                "energized": r.energized,
                "supercharged": r.supercharged,
                "traversal": r.traversal,
                "autoTowerLevels": r.auto_tower_levels,
                "endgameTowerLevels": r.endgame_tower_levels,
                "won": r.won,
                "tied": r.tied,
            }
            for r in state.results
        ],
        "remaining": [
            {
                "key": m.key,
                "number": m.number,
                "red": m.red,
                "blue": m.blue,
                "redCounting": m.counting("red"),
                "blueCounting": m.counting("blue"),
                "redSurrogates": sorted(m.red_surrogates),
                "blueSurrogates": sorted(m.blue_surrogates),
            }
            for m in state.remaining
        ],
        # Every constant the JS model needs, so the two implementations cannot
        # drift on a hardcoded number.
        "constants": {
            "winRp": WIN_RP,
            "tieRp": TIE_RP,
            "ridge": ridge,
            "autoTowerValue": AUTO_TOWER_VALUE,
            "endgameTowerValue": ENDGAME_TOWER_VALUE,
            "defaultThresholds": DEFAULT_THRESHOLDS,
            "towerPriorWeight": TOWER_PRIOR_WEIGHT,
            "defaultClimbTrust": DEFAULT_CLIMB_TRUST,
            "minClimbPrior": MIN_CLIMB_PRIOR,
            "tierOrder": TIER_ORDER,
        },
    }


def write(state: EventState, path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle(state, **kwargs), separators=(",", ":")))
    return path
