"""Wire the TBA client, the scouting export, the CSV standings and the model."""

from __future__ import annotations

from pathlib import Path

from .event import EventState, build_state
from .model import Fit, fit as fit_model
from .scouting import DEFAULT_URL as SCOUTING_URL, Scouting, ScoutingError
from .scouting import load as load_scouting_data
from .tba import CACHE_DIR, TBAClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENT = "2026auwarp"
DEFAULT_CSV = ROOT.parent / "event-rankings.csv"


def load_event(
    event_key: str = DEFAULT_EVENT,
    csv_path: Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> EventState:
    client = TBAClient(cache_dir=CACHE_DIR / event_key, offline=offline)
    try:
        event = client.event(event_key, force=refresh)
        name = event.get("name", event_key)
    except Exception:
        name = event_key
    matches = client.matches(event_key, force=refresh)
    csv_path = DEFAULT_CSV if csv_path is None else Path(csv_path)
    state = build_state(event_key, name, matches, csv_path)
    state.tba_source = client.last_source
    state.tba_warnings = list(client.warnings)
    return state


def load_scouting(
    state: EventState,
    url: str = SCOUTING_URL,
    offline: bool = False,
    refresh: bool = False,
) -> tuple[Scouting | None, str | None]:
    """Returns (scouting, error). A missing scouting export is not fatal --
    the simulator falls back to the on-field fit alone."""
    try:
        data = load_scouting_data(
            state,
            url=url,
            cache_dir=CACHE_DIR / state.event_key,
            offline=offline,
            refresh=refresh,
        )
    except ScoutingError as exc:
        return None, str(exc)
    return data, None


def load_fit(
    state: EventState,
    ridge: float = 3.0,
    scouting: Scouting | None = None,
    scouting_weight: float = 1.0,
) -> Fit:
    return fit_model(
        state, ridge=ridge, scouting=scouting, scouting_weight=scouting_weight
    )
