// Runs the browser model over the exported bundle and prints its fit as JSON,
// so tests/test_parity.py can diff it against the Python implementation.
//
//   node tests/js_fit.mjs <bundle.json> <scouting.json|-> [scoutingWeight]

import { readFileSync } from "node:fs";
import { fit, fitSummary } from "../web/lib/model.js";
import { build as buildScouting } from "../web/lib/scouting.js";
import { rankOrder, simulate } from "../web/lib/simulate.js";

const [bundlePath, scoutingPath, weightArg] = process.argv.slice(2);
const bundle = JSON.parse(readFileSync(bundlePath, "utf8"));

let scouting = null;
if (scoutingPath && scoutingPath !== "-") {
  scouting = buildScouting(JSON.parse(readFileSync(scoutingPath, "utf8")), {
    teams: bundle.teams,
    aliases: bundle.csvAliases,
    tierOrder: bundle.constants.tierOrder,
  });
}

const fitted = fit(bundle, {
  scouting,
  scoutingWeight: weightArg === undefined ? 1 : Number(weightArg),
});

// A short deterministic run, to check the sampler wires the fit up correctly.
const sim = simulate(bundle, fitted, { n: 4000, seed: 7, cutoff: 8 });

console.log(
  JSON.stringify({
    fit: fitSummary(fitted),
    scouting: scouting && {
      totalReports: scouting.totalReports,
      teamsCovered: scouting.teamsCovered,
      unmatched: scouting.unmatched,
      teams: scouting.teams,
    },
    currentOrder: rankOrder(bundle.standings, bundle.teams),
    sim: {
      teams: sim.teams.map((t) => ({
        team: t.team,
        meanRank: t.meanRank,
        pCutoff: t.pCutoff,
        expectedRp: t.expectedRp,
      })),
      matches: sim.matches.map((m) => ({ key: m.key, pRed: m.pRed, pBlue: m.pBlue })),
    },
  })
);
