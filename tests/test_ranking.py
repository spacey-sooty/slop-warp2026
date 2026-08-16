"""Checks that the ranking rules reproduce what the event actually published.

Run with:  python3 -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranksim.event import (  # noqa: E402
    TIE_RP,
    WIN_RP,
    published_mismatches,
    rank_teams,
    team_num,
)
from ranksim.loader import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_EVENT,
    load_event,
    load_fit,
    load_scouting,
)
from ranksim.scouting import TIER_ORDER, build as build_scouting  # noqa: E402
from ranksim.simulate import SimOptions, simulate  # noqa: E402

# Committed fixtures, so the suite is hermetic and CI needs no warm cache.
FIXTURES = ROOT / "tests" / "fixtures"
CACHE = FIXTURES / DEFAULT_EVENT


def load_state():
    return load_event(DEFAULT_EVENT, csv_path=DEFAULT_CSV, offline=True, cache_dir=FIXTURES)


class RankingRules(unittest.TestCase):
    def setUp(self):
        self.state = load_state()

    def test_standings_match_tba_rankings(self):
        """Standings rebuilt from the match breakdowns == TBA's own rankings."""
        published = json.loads((CACHE / "rankings.json").read_text())["rankings"]
        order = rank_teams(self.state.records)
        self.assertEqual(len(published), len(order))
        for row in published:
            team = team_num(row["team_key"])
            with self.subTest(team=team):
                self.assertEqual(order[row["rank"] - 1], team, "rank order")
                record = self.state.records[team]
                self.assertEqual(record.played, row["matches_played"])
                self.assertEqual(record.wins, row["record"]["wins"])
                self.assertEqual(record.losses, row["record"]["losses"])
                self.assertEqual(record.ties, row["record"]["ties"])
                self.assertEqual(record.rp, row["extra_stats"][0])
                for got, want in zip(record.sort_orders, row["sort_orders"]):
                    # TBA publishes the sort orders at single precision.
                    self.assertAlmostEqual(got, want, delta=1e-6 + abs(want) * 1e-6)

    def test_published_check_agrees(self):
        """The helper the CLI and the page both report from finds no fault."""
        self.assertIsNotNone(self.state.tba_rankings)
        self.assertEqual(published_mismatches(self.state.records, self.state.tba_rankings), [])

    def test_published_check_catches_a_wrong_rebuild(self):
        """...and would say so if the rules here drifted from the real ones."""
        broken = {t: r.copy() for t, r in self.state.records.items()}
        leader = rank_teams(broken)[0]
        broken[leader].rp -= 6
        mismatches = published_mismatches(broken, self.state.tba_rankings)
        self.assertTrue(mismatches)
        self.assertTrue(any(leader in m and "rp" in m for m in mismatches))

    def test_published_check_is_skipped_without_a_reference(self):
        self.assertIsNone(published_mismatches(self.state.records, None))

    def test_rp_formula(self):
        """3/1/0 plus one RP per achievement reproduces every breakdown's rp."""
        for result in self.state.results:
            expected = (
                (WIN_RP if result.won else TIE_RP if result.tied else 0)
                + result.energized
                + result.supercharged
                + result.traversal
            )
            self.assertEqual(expected, result.rp, f"{result.match_key} {result.color}")

    def test_csv_reconciles_with_tba(self):
        self.assertEqual(self.state.csv_discrepancies, [])
        self.assertEqual(self.state.csv_aliases, {"9982": "4788B"})

    def test_surrogates_do_not_count(self):
        surrogate_appearances = sum(
            len(m.red_surrogates) + len(m.blue_surrogates) for m in self.state.played
        )
        self.assertGreater(surrogate_appearances, 0, "event should have surrogates to test")
        for match in self.state.played:
            for color in ("red", "blue"):
                for team in match.alliance(color):
                    if team not in match.counting(color):
                        self.assertLess(
                            self.state.records[team].played,
                            sum(
                                team in m.alliance("red") + m.alliance("blue")
                                for m in self.state.played
                            ),
                        )


class Simulation(unittest.TestCase):
    def setUp(self):
        self.state = load_state()
        self.fit = load_fit(self.state)

    def test_distributions_are_proper(self):
        result = simulate(self.state, self.fit, SimOptions(n=500, seed=11))
        self.assertEqual(len(result["teams"]), len(self.state.teams))
        for row in result["teams"]:
            self.assertAlmostEqual(sum(row["rankDist"]), 1.0, places=6)
            self.assertGreaterEqual(row["meanRank"], 1.0)
        ranks_per_trial = [0.0] * len(self.state.teams)
        for row in result["teams"]:
            for i, p in enumerate(row["rankDist"]):
                ranks_per_trial[i] += p
        for total in ranks_per_trial:
            self.assertAlmostEqual(total, 1.0, places=6, msg="each rank taken exactly once")

    def test_seed_is_deterministic(self):
        a = simulate(self.state, self.fit, SimOptions(n=300, seed=5))
        b = simulate(self.state, self.fit, SimOptions(n=300, seed=5))
        self.assertEqual(a["teams"], b["teams"])

    def test_forcing_a_match_pins_its_outcome(self):
        match = self.state.remaining[0]
        result = simulate(
            self.state,
            self.fit,
            SimOptions(n=300, seed=5, forced={match.key: "blue"}),
        )
        row = next(m for m in result["matches"] if m["key"] == match.key)
        self.assertEqual(row["pBlue"], 1.0)
        self.assertEqual(row["pRed"], 0.0)

    def test_winning_out_cannot_hurt_a_team(self):
        team = self.state.teams[0]
        forced_win = {}
        for match in self.state.remaining:
            if team in match.red:
                forced_win[match.key] = "red"
            elif team in match.blue:
                forced_win[match.key] = "blue"
        base = simulate(self.state, self.fit, SimOptions(n=1500, seed=9))
        best = simulate(self.state, self.fit, SimOptions(n=1500, seed=9, forced=forced_win))

        def top8(res):
            return next(r["pCutoff"] for r in res["teams"] if r["team"] == team)

        self.assertGreaterEqual(top8(best) + 1e-9, top8(base))


class ScoutingIntegration(unittest.TestCase):
    def setUp(self):
        self.state = load_state()
        self.scouting, error = load_scouting(self.state, offline=True, cache_dir=FIXTURES)
        if self.scouting is None:
            self.skipTest(f"no cached scouting dump: {error}")

    def test_team_ids_reconcile(self):
        """The scouting app's 9982 is TBA's 4788B; both map onto the event."""
        covered = {t for t, s in self.scouting.teams.items() if s.reports}
        self.assertEqual(covered, set(self.state.teams))
        self.assertEqual(self.scouting.unmatched, ["9275 (2 reports)", "9975 (2 reports)"])

    def test_picklist_positions_are_overall_not_within_tier(self):
        """Consensus position must respect tier order, not raw in-tier rank."""
        ranked = {
            t: s.picklist_rank
            for t, s in self.scouting.teams.items()
            if s.picklist_rank is not None
        }
        self.assertGreater(len(ranked), 20)
        # Every team appears on ~27 lists, so positions span the whole field.
        self.assertGreater(max(ranked.values()), 15)
        # An S-tier team must place ahead of a D-tier one.
        s_tier = [t for t, s in self.scouting.teams.items() if s.picklist_tier == "S"]
        d_tier = [t for t, s in self.scouting.teams.items() if s.picklist_tier == "D"]
        self.assertTrue(s_tier and d_tier)
        self.assertLess(
            max(ranked[t] for t in s_tier), min(ranked[t] for t in d_tier)
        )

    def test_priors_are_calibrated_not_trusted_blindly(self):
        fit = load_fit(self.state, scouting=self.scouting)
        info = fit.scouting
        self.assertTrue(info["used"])
        # Trust is r^2-scaled, so it can never exceed the requested weight.
        self.assertLessEqual(info["hub"]["trust"], 1.0)
        self.assertAlmostEqual(info["hub"]["trust"], info["hub"]["rSquared"], places=6)

    def test_zero_weight_reproduces_the_plain_fit(self):
        plain = load_fit(self.state)
        zero = load_fit(self.state, scouting=self.scouting, scouting_weight=0.0)
        for team in self.state.teams:
            self.assertAlmostEqual(plain.hub[team], zero.hub[team], places=9)

    def test_climb_prior_never_rules_a_climb_impossible(self):
        fit = load_fit(self.state, scouting=self.scouting)
        for team in self.state.teams:
            self.assertGreater(fit.tower_rate(team), 0.0, team)

    def test_scouted_climber_outranks_the_rest(self):
        fit = load_fit(self.state, scouting=self.scouting)
        climbers = [t for t, s in self.scouting.teams.items() if s.can_climb_rate > 0.2]
        self.assertTrue(climbers)
        others = [t for t in self.state.teams if t not in climbers]
        self.assertGreater(
            min(fit.tower_rate(t) for t in climbers),
            max(fit.tower_rate(t) for t in others),
        )

    def test_missing_scouting_is_not_fatal(self):
        empty = build_scouting(self.state, {"event": {}, "pitReports": [], "picklists": []})
        fit = load_fit(self.state, scouting=empty)
        self.assertTrue(all(t in fit.hub for t in self.state.teams))

    def test_tier_order_matches_the_app(self):
        self.assertEqual(TIER_ORDER, ["S", "A", "B", "C", "D", "DNP"])


class TbaClient(unittest.TestCase):
    """Guards the two ways a refresh can quietly lie about its data."""

    def test_sends_a_user_agent(self):
        """TBA answers 403 to urllib's default UA, which made every live fetch
        fall back to cache while looking like it had succeeded."""
        from ranksim.tba import USER_AGENT

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"[]"

        def fake_urlopen(req, timeout=None):
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            return FakeResponse()

        import ranksim.tba as tba

        with unittest.mock.patch.object(tba.urllib.request, "urlopen", fake_urlopen):
            client = tba.TBAClient(api_key="k", cache_dir=Path(self.tmp))
            client.matches("2026auwarp", force=True)

        self.assertEqual(captured["headers"].get("user-agent"), USER_AGENT)
        self.assertEqual(captured["headers"].get("x-tba-auth-key"), "k")

    def test_reports_when_it_falls_back_to_cache(self):
        """A failed pull may serve cache -- it must not claim to be live."""
        import ranksim.tba as tba

        cache_dir = Path(self.tmp)
        (cache_dir / "matches.json").write_text("[1, 2, 3]")

        def boom(req, timeout=None):
            raise tba.urllib.error.URLError("no route to host")

        with unittest.mock.patch.object(tba.urllib.request, "urlopen", boom):
            client = tba.TBAClient(api_key="k", cache_dir=cache_dir)
            data = client.matches("2026auwarp", force=True)

        self.assertEqual(data, [1, 2, 3], "should still serve the cached copy")
        self.assertEqual(client.last_source, "stale-cache")
        self.assertTrue(client.warnings)
        self.assertIn("no route to host", client.warnings[0])

    def setUp(self):
        import tempfile
        import unittest.mock  # noqa: F401  (imported for the patches above)

        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()


class Bundle(unittest.TestCase):
    def test_bundle_records_its_provenance(self):
        from ranksim.export import bundle as build_bundle

        state = load_state()
        data = build_bundle(state)
        self.assertIn(data["tbaSource"], {"tba", "cache", "stale-cache"})
        self.assertEqual(len(data["results"]), len(state.results))
        self.assertEqual(len(data["remaining"]), len(state.remaining))
        self.assertEqual(set(data["standings"]), set(state.teams))

    def test_bundle_carries_what_a_rebuild_needs(self):
        """The page rebuilds the standings itself, so it needs the counting
        teams per appearance and TBA's table to check the result against."""
        from ranksim.export import bundle as build_bundle

        state = load_state()
        data = build_bundle(state)
        for row, result in zip(data["results"], state.results):
            self.assertEqual(row["countingTeams"], result.counting_teams)
        surrogate_rows = [
            r for r in data["results"] if len(r["countingTeams"]) < len(r["teams"])
        ]
        self.assertTrue(surrogate_rows, "event should have surrogates to test")
        self.assertEqual(data["tbaRankings"]["order"], rank_teams(state.records))


class StandingsSource(unittest.TestCase):
    """The two starting points a projection can be run from."""

    def setUp(self):
        self.state = load_state()
        self.fit = load_fit(self.state)

    def test_csv_and_tba_bases_are_both_usable(self):
        for source in ("csv", "tba"):
            with self.subTest(source=source):
                result = simulate(
                    self.state, self.fit, SimOptions(n=200, seed=4, source=source)
                )
                self.assertEqual(result["meta"]["source"], source)
                self.assertEqual(len(result["teams"]), len(self.state.teams))

    def test_tba_base_ignores_the_csv(self):
        """A stale CSV must not leak into the rebuilt-from-matches projection.

        The snapshot is fabricated from the match records rather than read from
        the CSV: that file lives outside the repo, so CI runs without one.
        """
        stale = {t: r.copy() for t, r in self.state.records.items()}
        for record in stale.values():
            record.rp += 50
        self.state.csv_records = stale

        from_csv = simulate(self.state, self.fit, SimOptions(n=200, seed=4, source="csv"))
        from_tba = simulate(self.state, self.fit, SimOptions(n=200, seed=4, source="tba"))
        csv_rp = {r["team"]: r["current"]["rp"] for r in from_csv["teams"]}
        tba_rp = {r["team"]: r["current"]["rp"] for r in from_tba["teams"]}
        for team, record in self.state.records.items():
            self.assertEqual(tba_rp[team], record.rp)
            self.assertEqual(csv_rp[team], record.rp + 50)


if __name__ == "__main__":
    unittest.main()
