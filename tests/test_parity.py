"""The browser model must agree with the Python model, number for number.

The static build reimplements the fit in JavaScript so the page can refit
against freshly pulled scouting without a server. Two implementations of the
same statistics is a real maintenance hazard -- this is the guard against them
drifting apart. If a change to ranksim/model.py is not mirrored in
web/lib/model.js, these tests fail.

Skipped if node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranksim.export import bundle as build_bundle  # noqa: E402
from ranksim.loader import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_EVENT,
    load_event,
    load_fit,
    load_scouting,
)
from ranksim.event import rank_teams  # noqa: E402

# Committed fixtures, so the suite is hermetic and CI needs no warm cache.
FIXTURES = ROOT / "tests" / "fixtures"
CACHE = FIXTURES / DEFAULT_EVENT
TOLERANCE = 1e-9


def js_available() -> bool:
    return shutil.which("node") is not None


class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not js_available():
            raise unittest.SkipTest("node not installed")
        cls.state = load_event(
            DEFAULT_EVENT, csv_path=DEFAULT_CSV, offline=True, cache_dir=FIXTURES
        )
        cls.scouting, _ = load_scouting(cls.state, offline=True, cache_dir=FIXTURES)
        cls.bundle = build_bundle(cls.state)

        # A real temp dir, not cache/ -- a clean checkout has no cache/ at all.
        cls._tmpdir = tempfile.TemporaryDirectory()
        bundle_path = Path(cls._tmpdir.name) / "bundle.json"
        bundle_path.write_text(json.dumps(cls.bundle))
        scouting_path = CACHE / "scouting.json"
        cls.bundle_path = bundle_path

        def run_js(scouting_arg: str, weight: str = "1") -> dict:
            proc = subprocess.run(
                [
                    "node",
                    str(ROOT / "tests" / "js_fit.mjs"),
                    str(bundle_path),
                    scouting_arg,
                    weight,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise AssertionError(f"js_fit.mjs failed:\n{proc.stderr}")
            return json.loads(proc.stdout)

        cls.js_plain = run_js("-")
        cls.js_scouted = run_js(str(scouting_path)) if scouting_path.exists() else None

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, "_tmpdir", None)
        if tmp is not None:
            tmp.cleanup()

    def assert_fit_matches(self, py_fit, js):
        summary = js["fit"]
        self.assertAlmostEqual(py_fit.sigma_hub, summary["sigmaHub"], delta=TOLERANCE)
        self.assertAlmostEqual(py_fit.sigma_auto, summary["sigmaAuto"], delta=TOLERANCE)
        self.assertAlmostEqual(py_fit.r_squared, summary["rSquared"], delta=TOLERANCE)
        self.assertEqual(py_fit.n_observations, summary["observations"])

        for name, value in py_fit.thresholds.items():
            with self.subTest(threshold=name):
                self.assertAlmostEqual(value, summary["thresholds"][name], delta=TOLERANCE)
                self.assertEqual(
                    py_fit.threshold_sources[name], summary["thresholdSources"][name]
                )

        for team in py_fit.teams:
            with self.subTest(team=team):
                got = summary["teams"][team]
                self.assertAlmostEqual(py_fit.hub[team], got["hubOpr"], delta=TOLERANCE)
                self.assertAlmostEqual(py_fit.hub_se[team], got["hubSe"], delta=TOLERANCE)
                self.assertAlmostEqual(py_fit.auto[team], got["autoOpr"], delta=TOLERANCE)
                self.assertAlmostEqual(
                    py_fit.tower_rate(team), got["towerRate"], delta=TOLERANCE
                )
                self.assertAlmostEqual(
                    py_fit.expected_tower(team), got["expectedTower"], delta=TOLERANCE
                )

    def test_plain_fit_matches(self):
        self.assert_fit_matches(load_fit(self.state), self.js_plain)

    def test_scouted_fit_matches(self):
        if self.js_scouted is None:
            self.skipTest("no cached scouting dump")
        py_fit = load_fit(self.state, scouting=self.scouting)
        self.assert_fit_matches(py_fit, self.js_scouted)

    def test_scouting_reduction_matches(self):
        """Medians, rates, tags and picklist consensus must agree exactly."""
        if self.js_scouted is None:
            self.skipTest("no cached scouting dump")
        js = self.js_scouted["scouting"]
        self.assertEqual(js["totalReports"], self.scouting.total_reports)
        self.assertEqual(js["unmatched"], self.scouting.unmatched)
        for team, py in self.scouting.teams.items():
            with self.subTest(team=team):
                got = js["teams"][team]
                self.assertEqual(py.reports, got["reports"])
                self.assertAlmostEqual(py.balls_per_match, got["ballsPerMatch"], delta=TOLERANCE)
                self.assertAlmostEqual(py.auto_balls, got["autoBalls"], delta=TOLERANCE)
                self.assertAlmostEqual(py.driver_rating, got["driverRating"], delta=TOLERANCE)
                self.assertAlmostEqual(py.defense_rating, got["defenseRating"], delta=TOLERANCE)
                self.assertAlmostEqual(py.can_climb_rate, got["canClimbRate"], delta=TOLERANCE)
                self.assertEqual([list(t) for t in py.tags], got["tags"])
                self.assertEqual(py.notes, got["notes"])
                self.assertEqual(py.picklist_tier, got["picklistTier"])
                self.assertEqual(py.primary_rank, got["primaryRank"])
                if py.picklist_rank is None:
                    self.assertIsNone(got["picklistRank"])
                else:
                    self.assertAlmostEqual(
                        py.picklist_rank, got["picklistRank"], delta=TOLERANCE
                    )

    def test_scouting_calibration_matches(self):
        if self.js_scouted is None:
            self.skipTest("no cached scouting dump")
        py_info = load_fit(self.state, scouting=self.scouting).scouting
        js_info = self.js_scouted["fit"]["scouting"]
        self.assertTrue(js_info["used"])
        for key in ("hub", "auto"):
            with self.subTest(component=key):
                for field in ("rSquared", "trust", "slope", "intercept"):
                    self.assertAlmostEqual(
                        py_info[key][field], js_info[key][field], delta=TOLERANCE
                    )
        self.assertEqual(py_info["climb"]["mode"], js_info["climb"]["mode"])

    def test_current_ranking_order_matches(self):
        base = self.state.csv_records or self.state.records
        self.assertEqual(rank_teams(base), self.js_plain["currentOrder"])

    def test_sampler_agrees_statistically(self):
        """Different RNGs, so compare distributions rather than draws."""
        if self.js_scouted is None:
            self.skipTest("no cached scouting dump")
        from ranksim.simulate import SimOptions, simulate

        py = simulate(
            self.state,
            load_fit(self.state, scouting=self.scouting),
            SimOptions(n=20000, seed=3, cutoff=8),
        )
        py_rows = {r["team"]: r for r in py["teams"]}
        js_rows = {r["team"]: r for r in self.js_scouted["sim"]["teams"]}

        for team, py_row in py_rows.items():
            with self.subTest(team=team):
                js_row = js_rows[team]
                # 4k vs 20k trials of the same process: mean rank should land
                # within a third of a place, top-8 odds within 5 points.
                self.assertAlmostEqual(py_row["meanRank"], js_row["meanRank"], delta=0.34)
                self.assertAlmostEqual(py_row["pCutoff"], js_row["pCutoff"], delta=0.05)
                self.assertAlmostEqual(py_row["expectedRp"], js_row["expectedRp"], delta=0.6)

        py_matches = {m["key"]: m for m in py["matches"]}
        for m in self.js_scouted["sim"]["matches"]:
            with self.subTest(match=m["key"]):
                self.assertAlmostEqual(py_matches[m["key"]]["pRed"], m["pRed"], delta=0.05)


if __name__ == "__main__":
    unittest.main()
