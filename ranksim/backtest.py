"""Walk-forward backtest: how well does the model call matches it has not seen?

For each cut point, the event is rebuilt as if only the first `cut` qualification
matches had been played, the model is refitted on those alone, and the very next
match is predicted. The prediction is produced by the simulator itself, so what
is measured is the thing that actually drives the projections -- ratings, noise
model, climbs, fouls and all -- rather than a simplified stand-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .event import build_state
from .model import DEFAULT_HALF_LIFE, fit as fit_model
from .simulate import SimOptions, simulate


@dataclass
class Prediction:
    match_number: int
    p_red: float
    p_blue: float
    actual: str
    red_score: int
    blue_score: int
    predicted_red: float
    predicted_blue: float

    @property
    def called(self) -> bool:
        if self.actual == "":
            return False
        return (self.p_red > self.p_blue) == (self.actual == "red")

    @property
    def p_actual(self) -> float:
        return self.p_red if self.actual == "red" else self.p_blue


def _truncate(state, keep: int):
    """Rebuild the event with only the first `keep` played matches known."""
    played_numbers = sorted(m.number for m in state.played)
    horizon = played_numbers[keep - 1]
    raw = []
    for match in state.raw_matches:
        if match["match_number"] > horizon and match.get("score_breakdown"):
            match = {**match, "score_breakdown": None, "winning_alliance": ""}
        raw.append(match)
    return build_state(state.event_key, state.event_name, raw, None)


def run(
    state,
    ridge: float = 3.0,
    min_matches: int = 15,
    trials: int = 800,
    scouting=None,
    scouting_weight: float = 1.0,
    half_life: float = DEFAULT_HALF_LIFE,
) -> dict:
    played = sorted(state.played, key=lambda m: m.number)
    by_number = {m.number: m for m in state.matches}
    breakdowns = {
        r.match_key: r for r in state.results if r.color == "red"
    }
    blue_breakdowns = {r.match_key: r for r in state.results if r.color == "blue"}

    predictions: list[Prediction] = []
    for cut in range(min_matches, len(played)):
        target = played[cut]
        sub = _truncate(state, cut)
        # Scouting is filed in the pits before qualifications, so feeding the
        # full dump to an earlier cut point is not hindsight -- unlike the match
        # results, which are truncated.
        sub_fit = fit_model(
            sub,
            ridge=ridge,
            scouting=scouting,
            scouting_weight=scouting_weight,
            half_life=half_life,
        )
        result = simulate(
            sub,
            sub_fit,
            SimOptions(n=trials, seed=1234 + cut, cutoff=8, source="tba"),
        )
        row = next((m for m in result["matches"] if m["key"] == target.key), None)
        if row is None:
            continue
        red = breakdowns[target.key]
        blue = blue_breakdowns[target.key]
        predictions.append(
            Prediction(
                match_number=target.number,
                p_red=row["pRed"],
                p_blue=row["pBlue"],
                actual=by_number[target.number].winner,
                red_score=red.match_points,
                blue_score=blue.match_points,
                predicted_red=row["redPredicted"],
                predicted_blue=row["bluePredicted"],
            )
        )

    decided = [p for p in predictions if p.actual]
    n = len(decided)
    accuracy = sum(p.called for p in decided) / n if n else 0.0
    brier = sum((p.p_actual - 1.0) ** 2 for p in decided) / n if n else 0.0
    log_loss = (
        -sum(math.log(max(p.p_actual, 1e-6)) for p in decided) / n if n else 0.0
    )
    score_err = [
        abs(p.predicted_red - p.red_score) for p in decided
    ] + [abs(p.predicted_blue - p.blue_score) for p in decided]
    mae = sum(score_err) / len(score_err) if score_err else 0.0

    # Calibration: of the matches we called at X% confidence, how many landed?
    buckets = []
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        rows = [p for p in decided if lo <= max(p.p_red, p.p_blue) < hi]
        if rows:
            buckets.append(
                {
                    "range": f"{int(lo * 100)}-{int(hi * 100 if hi <= 1 else 100)}%",
                    "n": len(rows),
                    "predicted": sum(max(p.p_red, p.p_blue) for p in rows) / len(rows),
                    "actual": sum(p.called for p in rows) / len(rows),
                }
            )

    return {
        "predictions": predictions,
        "n": n,
        "accuracy": accuracy,
        "brier": brier,
        "logLoss": log_loss,
        "scoreMae": mae,
        "calibration": buckets,
        # A coin flip scores 0.25; always guessing the favourite at the observed
        # base rate is the other reference point worth beating.
        "brierBaseline": 0.25,
    }
