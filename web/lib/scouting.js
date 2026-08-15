// Fetch the deployed scouting export and reduce it to per-team figures.
//
// Port of ranksim/scouting.py. This runs in the browser against the live
// deployment -- the app's /api/scouting route sets Access-Control-Allow-Origin:*
// precisely so browser-side tools can read it, and it needs no key.
//
// Per-team figures are medians across scouts: one report claiming 200 balls per
// match should not move a team.

import { mean, median } from "./linalg.js";

export async function fetchScouting(url, eventKey, { signal } = {}) {
  const target = eventKey ? `${url}?event=${encodeURIComponent(eventKey)}` : url;
  // The export is a live snapshot and serves no-store; cache: "no-store" keeps
  // the browser from handing back a stale copy on the refresh button.
  const res = await fetch(target, { signal, cache: "no-store" });
  if (!res.ok) throw new Error(`scouting export returned ${res.status} ${res.statusText}`);
  const payload = await res.json();
  if (payload.error) throw new Error(payload.error);
  return payload;
}

const rate = (values) =>
  values.length ? values.filter(Boolean).length / values.length : 0;

export function build(raw, { teams, aliases = {}, tierOrder, maxNotes = 6 }) {
  const known = new Set(teams);
  const resolve = (number) => {
    const key = String(number);
    return aliases[key] !== undefined ? aliases[key] : key;
  };

  const byTeam = {};
  const unmatched = new Map();
  for (const report of raw.pitReports || []) {
    if (report.teamNumber === null || report.teamNumber === undefined) continue;
    const key = resolve(report.teamNumber);
    if (!known.has(key)) {
      unmatched.set(String(report.teamNumber), (unmatched.get(String(report.teamNumber)) || 0) + 1);
      continue;
    }
    (byTeam[key] = byTeam[key] || []).push(report);
  }

  // Picklists are tiered and an entry's `rank` is its position *within* its
  // tier, so a D-tier and an S-tier team both hold rank 0. Entries have to be
  // flattened to an overall position before anything is averaged.
  const ordered = (entries) => {
    const rows = [];
    for (const entry of entries || []) {
      if (entry.teamNumber === null || entry.teamNumber === undefined) continue;
      const key = resolve(entry.teamNumber);
      if (known.has(key)) rows.push([key, entry]);
    }
    rows.sort((a, b) => {
      const ta = tierOrder.indexOf(a[1].tier);
      const tb = tierOrder.indexOf(b[1].tier);
      const ra = ta === -1 ? tierOrder.length : ta;
      const rb = tb === -1 ? tierOrder.length : tb;
      return ra - rb || a[1].rank - b[1].rank;
    });
    return rows;
  };

  const personalPositions = {};
  const personalTiers = {};
  const primary = {};
  for (const plist of raw.picklists || []) {
    ordered(plist.entries).forEach(([key, entry], i) => {
      const position = i + 1;
      if (plist.kind === "primary") {
        primary[key] = { rank: position, tier: entry.tier };
      } else {
        (personalPositions[key] = personalPositions[key] || []).push(position);
        (personalTiers[key] = personalTiers[key] || []).push(entry.tier);
      }
    });
  }

  const teamsOut = {};
  for (const team of teams) {
    const reports = byTeam[team] || [];
    const s = {
      team,
      reports: reports.length,
      ballsPerMatch: 0,
      autoBalls: 0,
      storage: 0,
      driverRating: 0,
      defenseRating: 0,
      canClimbRate: 0,
      autoClimbRate: 0,
      canScoreRate: 0,
      hasAutoRate: 0,
      tags: [],
      notes: [],
      picklistRank: null,
      picklistLists: 0,
      picklistTier: null,
      primaryTier: null,
      primaryRank: null,
    };

    if (reports.length) {
      const defined = (field) =>
        reports.map((r) => r[field]).filter((v) => v !== null && v !== undefined);
      s.ballsPerMatch = median(defined("ballsPerMatch"));
      s.autoBalls = median(defined("autoBalls"));
      s.storage = median(defined("storageCapacity"));
      s.driverRating = mean(reports.map((r) => r.driverRating));
      s.defenseRating = mean(reports.map((r) => r.defenseRating));
      s.canClimbRate = rate(reports.map((r) => Boolean(r.canClimb)));
      s.autoClimbRate = rate(defined("autoClimb").map(Boolean));
      s.canScoreRate = rate(reports.map((r) => Boolean(r.canScoreBalls)));
      s.hasAutoRate = rate(reports.map((r) => Boolean(r.hasAuto)));

      const tagCounts = new Map();
      for (const r of reports) {
        for (const tag of r.tags || []) {
          const key = tag.trim().toLowerCase();
          tagCounts.set(key, (tagCounts.get(key) || 0) + 1);
        }
      }
      s.tags = [...tagCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
      s.notes = reports
        .map((r) => (r.notes || "").trim())
        .filter(Boolean)
        .slice(0, maxNotes);
    }

    const positions = personalPositions[team] || [];
    if (positions.length) {
      s.picklistRank = mean(positions);
      s.picklistLists = positions.length;
      const tiers = personalTiers[team] || [];
      if (tiers.length) {
        const counts = new Map();
        for (const t of tiers) counts.set(t, (counts.get(t) || 0) + 1);
        s.picklistTier = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
      }
    }
    if (primary[team]) {
      s.primaryRank = primary[team].rank;
      s.primaryTier = primary[team].tier;
    }
    teamsOut[team] = s;
  }

  return {
    eventKey: (raw.event || {}).tbaKey || "",
    eventName: (raw.event || {}).name || "",
    teams: teamsOut,
    unmatched: [...unmatched.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([team, n]) => `${team} (${n} reports)`),
    totalReports: (raw.pitReports || []).length,
    picklists: (raw.picklists || []).length,
    teamsCovered: Object.values(teamsOut).filter((t) => t.reports).length,
    exportedAt: raw.exportedAt || 0,
    fetchedAt: Date.now(),
  };
}
