// Unit-tests web/lib/standings.js against a hand-built event whose answers can
// be worked out on paper, so the browser's rebuild of the ranking rules is
// checked on its own rather than only through the 2026auwarp fixture.
//
//   node tests/js_standings.mjs        exits non-zero, printing failures
//
// tests/test_ranking.py runs this.

import {
  buildStandings,
  diffStandings,
  verifyAgainstPublished,
} from "../web/lib/standings.js";
import { rankOrder } from "../web/lib/simulate.js";

const failures = [];
const check = (name, got, want) => {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g !== w) failures.push(`${name}: got ${g}, want ${w}`);
};

// Two matches. 1/2/3 beat 4/5/6 with both hub achievements; then 1/2/4 lose to
// 3/5/6, where 4 is a surrogate and must score for the alliance but not for
// itself. RP: 3 for a win, 1 for a tie, plus one per achievement.
const result = (n, color, teams, counting, extra) => ({
  matchKey: `x_qm${n}`,
  matchNumber: n,
  color,
  teams,
  countingTeams: counting,
  hubPoints: extra.hub,
  hubAuto: extra.auto,
  towerPoints: extra.tower,
  foulPoints: 0,
  matchPoints: extra.hub + extra.tower,
  rp: extra.rp,
  energized: true,
  supercharged: false,
  traversal: false,
  autoTowerLevels: ["None", "None", "None"],
  endgameTowerLevels: ["None", "None", "None"],
  won: extra.won,
  tied: false,
});

const bundle = {
  teams: ["1", "2", "3", "4", "5", "6"],
  results: [
    result(1, "red", ["1", "2", "3"], ["1", "2", "3"], {
      hub: 120, auto: 30, tower: 10, rp: 5, won: true,
    }),
    result(1, "blue", ["4", "5", "6"], ["4", "5", "6"], {
      hub: 60, auto: 10, tower: 0, rp: 1, won: false,
    }),
    result(2, "red", ["1", "2", "4"], ["1", "2"], {
      hub: 40, auto: 5, tower: 0, rp: 1, won: false,
    }),
    result(2, "blue", ["3", "5", "6"], ["3", "5", "6"], {
      hub: 90, auto: 20, tower: 20, rp: 4, won: true,
    }),
  ],
};

const standings = buildStandings(bundle);

// 4 played only match 1: its match-2 appearance was a surrogate one.
check("surrogate does not count", standings["4"].played, 1);
check("surrogate keeps its real match", standings["4"].rp, 1);
check("teammate of the surrogate counts both", standings["1"].played, 2);

check("rp accumulates", standings["1"].rp, 6);
check("wins/losses", [standings["1"].wins, standings["1"].losses], [1, 1]);
check("match points exclude fouls", standings["1"].matchPoints, 130 + 40);
check("auto fuel", standings["1"].autoFuel, 35);
check("tower", standings["1"].tower, 10);
check("ranking score is the per-match average", standings["1"].rankScore, 3);
check("avg match", standings["3"].avgMatch, (130 + 110) / 2);

// Ranking score is RP per match, so 3 (9/2) leads, then 1 and 2 (6/2, split on
// team number), then 5 and 6 (5/2), then the surrogate-hit 4 (1/1).
check("order", rankOrder(standings, bundle.teams), ["3", "1", "2", "5", "6", "4"]);

// A snapshot that missed match 2 entirely -- the case the button exists for.
const stale = buildStandings({ teams: bundle.teams, results: bundle.results.slice(0, 2) });
const diff = diffStandings(stale, standings, bundle.teams);
check("stale snapshot differs for everyone who played match 2", diff.changed.length, 5);
check(
  "and the ranking moves",
  diff.moved.length > 0 && diff.order[0] === "3",
  true
);
check("identical standings diff to nothing", diffStandings(standings, standings, bundle.teams).changed.length, 0);

// Verification against a published table.
const published = {
  order: ["3", "1", "2", "5", "6", "4"],
  teams: {
    3: { rank: 1, played: 2, rp: 9, wins: 2, losses: 0, ties: 0 },
    1: { rank: 2, played: 2, rp: 6, wins: 1, losses: 1, ties: 0 },
  },
};
check("agreeing table verifies", verifyAgainstPublished(standings, published).agrees, true);
check("no table means nothing to check", verifyAgainstPublished(standings, null).checked, false);

const wrong = { ...published, teams: { ...published.teams, 1: { ...published.teams[1], rp: 7 } } };
const bad = verifyAgainstPublished(standings, wrong);
check("a disagreement is reported", [bad.checked, bad.agrees], [true, false]);
check("and says what disagrees", bad.mismatches, ["1 rp: TBA 7, rebuild 6"]);

const reordered = { order: ["1", "3", "2", "5", "6", "4"], teams: {} };
check(
  "a different published order is reported too",
  verifyAgainstPublished(standings, reordered).mismatches.length,
  2
);

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(JSON.stringify({ ok: true, checks: 17 }));
