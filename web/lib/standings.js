// Rebuild the current standings from the played qualification matches.
//
// Port of the accumulation in ranksim/event.py's build_state, so the page can
// start a projection from what the match results actually say rather than from
// whatever standings were exported with the bundle. The CSV that ships as the
// starting point is a snapshot taken at some moment; TBA's match record is not,
// and mid-event the two drift apart the instant a match is scored.
//
// The ranking rules are the 2026 ones (see ranksim/event.py): 3/1/0 for the
// match plus one RP each for Energized, Supercharged and Traversal, ranked by
// ranking score then average match points, auto fuel and tower -- all averages
// over counting matches, with surrogate appearances excluded from the
// surrogate's own record.
//
// TBA publishes its own table computed from the same matches. It is not the
// input here -- standings have to be rebuildable to be projectable -- but it is
// carried in the bundle as `tbaRankings` and `verifyAgainstPublished` below
// checks this rebuild against it, so a disagreement surfaces instead of
// quietly seeding every projection from a wrong table.

import { rankOrder } from "./simulate.js";

const blank = (team) => ({
  team,
  rp: 0,
  matchPoints: 0,
  autoFuel: 0,
  tower: 0,
  wins: 0,
  losses: 0,
  ties: 0,
  played: 0,
});

// The fields TeamRecord.as_dict() derives, so a rebuilt record is
// interchangeable with an exported one everywhere downstream.
function withAverages(record) {
  const n = record.played || 1;
  record.rankScore = record.rp / n;
  record.avgMatch = record.matchPoints / n;
  record.avgAuto = record.autoFuel / n;
  record.avgTower = record.tower / n;
  return record;
}

export function buildStandings(bundle) {
  const records = {};
  for (const team of bundle.teams) records[team] = blank(team);

  for (const result of bundle.results) {
    for (const team of result.countingTeams) {
      const record = records[team] || (records[team] = blank(team));
      record.played += 1;
      record.rp += result.rp;
      record.matchPoints += result.matchPoints;
      record.autoFuel += result.hubAuto;
      record.tower += result.towerPoints;
      if (result.won) record.wins += 1;
      else if (result.tied) record.ties += 1;
      else record.losses += 1;
    }
  }

  for (const record of Object.values(records)) withAverages(record);
  return records;
}

const COMPARED = [
  ["played", "played"],
  ["rp", "RP"],
  ["matchPoints", "match points"],
  ["autoFuel", "auto fuel"],
  ["tower", "tower"],
  ["wins", "W"],
  ["losses", "L"],
  ["ties", "T"],
];

// What switching between two sets of standings actually changes: which teams'
// records differ, and how far the current ranking moves.
export function diffStandings(from, to, teams) {
  const changed = [];
  for (const team of teams) {
    const a = from[team];
    const b = to[team];
    if (!a || !b) continue;
    const fields = COMPARED.filter(([key]) => a[key] !== b[key]).map(
      ([key, label]) => `${label} ${a[key]} → ${b[key]}`
    );
    if (fields.length) changed.push({ team, fields });
  }

  const before = rankOrder(from, teams);
  const after = rankOrder(to, teams);
  const moved = teams.filter((t) => before.indexOf(t) !== after.indexOf(t));
  return { changed, moved, order: after };
}

// Check a rebuild against TBA's published table. Only the fields TBA publishes
// are compared, and only for teams it lists.
export function verifyAgainstPublished(standings, published) {
  if (!published || !published.order || !published.order.length) {
    return { checked: false, agrees: false, mismatches: [] };
  }
  const mismatches = [];
  const order = rankOrder(standings, published.order);
  published.order.forEach((team, i) => {
    if (order[i] !== team) {
      mismatches.push(`rank ${i + 1}: TBA has ${team}, rebuild has ${order[i]}`);
    }
  });
  for (const [team, row] of Object.entries(published.teams || {})) {
    const record = standings[team];
    if (!record) {
      mismatches.push(`${team}: ranked by TBA but not in the rebuild`);
      continue;
    }
    for (const [key, published_] of [
      ["played", row.played],
      ["rp", row.rp],
      ["wins", row.wins],
      ["losses", row.losses],
      ["ties", row.ties],
    ]) {
      if (published_ !== null && published_ !== undefined && record[key] !== published_) {
        mismatches.push(`${team} ${key}: TBA ${published_}, rebuild ${record[key]}`);
      }
    }
  }
  return { checked: true, agrees: mismatches.length === 0, mismatches };
}
