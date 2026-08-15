// Monte Carlo over the remaining qual matches, in the browser.
//
// Port of ranksim/simulate.py. Returns the exact shape the Python /api/simulate
// returned, so the rendering code did not have to change.
//
// The RNG differs from Python's Mersenne Twister, so a given seed does not
// reproduce Python's draws -- seeds are reproducible within this build only.
// The fit is what tests/test_parity.py pins exactly; the sampler is checked
// statistically.

const FORCE_MAX_TRIES = 60;
const TRUNCATE_MAX_TRIES = 16;

// mulberry32: small, fast, and seedable, which the browser's Math.random is not.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFactory(random) {
  let spare = null;
  return function gauss(mu, sigma) {
    if (spare !== null) {
      const value = mu + sigma * spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = random() * 2 - 1;
      v = random() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const factor = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * factor;
    return mu + sigma * u * factor;
  };
}

function percentileFromCounts(counts, total, q) {
  const target = q * total;
  let seen = 0;
  for (let rank = 0; rank < counts.length; rank++) {
    seen += counts[rank];
    if (seen >= target) return rank + 1;
  }
  return counts.length;
}

function teamSortKey(team) {
  const digits = team.replace(/\D/g, "");
  return [digits ? Number(digits) : 0, team];
}

// The official tiebreaker chain: ranking score, then avg match, avg auto fuel,
// avg tower -- all averages over counting matches.
export function rankOrder(standings, teams) {
  return [...teams].sort((a, b) => {
    const ra = standings[a];
    const rb = standings[b];
    const na = ra.played || 1;
    const nb = rb.played || 1;
    const keys = [
      [rb.rp / nb, ra.rp / na],
      [rb.matchPoints / nb, ra.matchPoints / na],
      [rb.autoFuel / nb, ra.autoFuel / na],
      [rb.tower / nb, ra.tower / na],
    ];
    for (const [x, y] of keys) if (x !== y) return x - y;
    const [an, as] = teamSortKey(a);
    const [bn, bs] = teamSortKey(b);
    return an - bn || as.localeCompare(bs);
  });
}

export function simulate(bundle, fit, options = {}) {
  const {
    n = 5000,
    seed = null,
    oprUncertainty = true,
    cutoff = 8,
    forced = {},
    useScouting = true,
  } = options;

  const constants = bundle.constants;
  const teams = bundle.teams;
  const nTeams = teams.length;
  const index = {};
  teams.forEach((t, i) => (index[t] = i));

  const random = mulberry32(seed === null ? (Math.random() * 4294967296) >>> 0 : seed);
  const gauss = gaussianFactory(random);

  const base = bundle.standings;
  const baseRp = teams.map((t) => base[t].rp);
  const baseMp = teams.map((t) => base[t].matchPoints);
  const baseAf = teams.map((t) => base[t].autoFuel);
  const baseTw = teams.map((t) => base[t].tower);
  const basePlayed = teams.map((t) => base[t].played);

  const hubRating = teams.map((t) => fit.hub[t]);
  const hubSe = teams.map((t) => fit.hubSe[t]);
  const autoRating = teams.map((t) => fit.auto[t]);
  const autoSe = teams.map((t) => fit.autoSe[t]);

  // Per-team climb sampler: cumulative weights over point outcomes.
  const towerPoints = [];
  const towerCum = [];
  for (const t of teams) {
    const outcomes = fit.towerOutcomes[t];
    const pts = [];
    const cum = [];
    let running = 0;
    for (const [outcome, prob] of outcomes) {
      running += prob;
      pts.push(outcome.points);
      cum.push(running);
    }
    if (cum.length) cum[cum.length - 1] = 1;
    else {
      pts.push(0);
      cum.push(1);
    }
    towerPoints.push(pts);
    towerCum.push(cum);
  }

  const fouls = fit.foulSamples.length ? fit.foulSamples : [0];
  const thrEnergized = fit.thresholds.energized;
  const thrSupercharged = fit.thresholds.supercharged;
  const thrTraversal = fit.thresholds.traversal;
  const sigmaHub = fit.sigmaHub;
  const sigmaAuto = fit.sigmaAuto;
  const winRp = constants.winRp;
  const tieRp = constants.tieRp;

  const sched = bundle.remaining.map((m) => ({
    key: m.key,
    red: m.red.map((t) => index[t]),
    blue: m.blue.map((t) => index[t]),
    redCounting: m.redCounting.map((t) => index[t]),
    blueCounting: m.blueCounting.map((t) => index[t]),
    force: forced[m.key] || "",
  }));

  const rankCounts = teams.map(() => new Int32Array(nTeams));
  const rankSum = new Float64Array(nTeams);
  const rpSum = new Float64Array(nTeams);
  const rsSum = new Float64Array(nTeams);
  const matchCounts = {};
  for (const m of bundle.remaining) matchCounts[m.key] = { red: 0, blue: 0, tie: 0 };
  let forcedNudges = 0;

  // Integer draw from Normal(mean, sigma) truncated at zero. Clamping instead of
  // truncating piles an atom onto exactly 0, which shows up as spurious ties.
  const drawScore = (m, sigma) => {
    for (let i = 0; i < TRUNCATE_MAX_TRIES; i++) {
      const value = m + gauss(0, sigma);
      if (value >= 0) return Math.floor(value + 0.5);
    }
    return 0;
  };

  const drawTower = (idx) => {
    const r = random();
    const cum = towerCum[idx];
    for (let i = 0; i < cum.length; i++) if (r <= cum[i]) return towerPoints[idx][i];
    return towerPoints[idx][cum.length - 1];
  };

  const rp = new Float64Array(nTeams);
  const mp = new Float64Array(nTeams);
  const af = new Float64Array(nTeams);
  const tw = new Float64Array(nTeams);
  const played = new Float64Array(nTeams);
  const order = new Int32Array(nTeams);
  const hubR = new Float64Array(nTeams);
  const autoR = new Float64Array(nTeams);

  for (let trial = 0; trial < n; trial++) {
    for (let i = 0; i < nTeams; i++) {
      hubR[i] = oprUncertainty ? hubRating[i] + gauss(0, hubSe[i]) : hubRating[i];
      autoR[i] = oprUncertainty ? autoRating[i] + gauss(0, autoSe[i]) : autoRating[i];
      rp[i] = baseRp[i];
      mp[i] = baseMp[i];
      af[i] = baseAf[i];
      tw[i] = baseTw[i];
      played[i] = basePlayed[i];
    }

    for (const match of sched) {
      const { red, blue, force } = match;
      let rHub = 0;
      let bHub = 0;
      let rAuto = 0;
      let bAuto = 0;
      let rTower = 0;
      let bTower = 0;
      let rTotal = 0;
      let bTotal = 0;
      let landed = false;

      const tries = force === "red" || force === "blue" ? FORCE_MAX_TRIES : 1;
      for (let attempt = 0; attempt < tries; attempt++) {
        let rMean = 0;
        let bMean = 0;
        let rAutoMean = 0;
        let bAutoMean = 0;
        for (let i = 0; i < 3; i++) {
          rMean += hubR[red[i]];
          bMean += hubR[blue[i]];
          rAutoMean += autoR[red[i]];
          bAutoMean += autoR[blue[i]];
        }
        rHub = drawScore(rMean, sigmaHub);
        bHub = drawScore(bMean, sigmaHub);
        rAuto = drawScore(rAutoMean, sigmaAuto);
        bAuto = drawScore(bAutoMean, sigmaAuto);
        if (rAuto > rHub) rAuto = rHub;
        if (bAuto > bHub) bAuto = bHub;
        rTower = drawTower(red[0]) + drawTower(red[1]) + drawTower(red[2]);
        bTower = drawTower(blue[0]) + drawTower(blue[1]) + drawTower(blue[2]);
        rTotal = rHub + rTower + fouls[(random() * fouls.length) | 0];
        bTotal = bHub + bTower + fouls[(random() * fouls.length) | 0];
        if (!force || force === "tie") {
          landed = true;
          break;
        }
        if ((force === "red" && rTotal > bTotal) || (force === "blue" && bTotal > rTotal)) {
          landed = true;
          break;
        }
      }
      if (!landed && (force === "red" || force === "blue")) {
        // Never landed the requested upset; push the forced winner over the line
        // by the smallest amount that flips it.
        forcedNudges++;
        if (force === "red") {
          const bump = bTotal - rTotal + 1;
          rHub += bump;
          rTotal += bump;
        } else {
          const bump = rTotal - bTotal + 1;
          bHub += bump;
          bTotal += bump;
        }
      }

      let redWin;
      let blueWin;
      let tie;
      if (force === "tie") {
        redWin = false;
        blueWin = false;
        tie = true;
      } else {
        redWin = rTotal > bTotal;
        blueWin = bTotal > rTotal;
        tie = !redWin && !blueWin;
      }

      const counts = matchCounts[match.key];
      if (redWin) counts.red++;
      else if (blueWin) counts.blue++;
      else counts.tie++;

      let rRp = redWin ? winRp : tie ? tieRp : 0;
      let bRp = blueWin ? winRp : tie ? tieRp : 0;
      if (rHub >= thrEnergized) {
        rRp += 1;
        if (rHub >= thrSupercharged) rRp += 1;
      }
      if (bHub >= thrEnergized) {
        bRp += 1;
        if (bHub >= thrSupercharged) bRp += 1;
      }
      if (rTower >= thrTraversal) rRp += 1;
      if (bTower >= thrTraversal) bRp += 1;

      const rMp = rHub + rTower;
      const bMp = bHub + bTower;
      for (const i of match.redCounting) {
        rp[i] += rRp;
        mp[i] += rMp;
        af[i] += rAuto;
        tw[i] += rTower;
        played[i] += 1;
      }
      for (const i of match.blueCounting) {
        rp[i] += bRp;
        mp[i] += bMp;
        af[i] += bAuto;
        tw[i] += bTower;
        played[i] += 1;
      }
    }

    for (let i = 0; i < nTeams; i++) order[i] = i;
    order.sort((a, b) => {
      const pa = played[a] || 1;
      const pb = played[b] || 1;
      let d = rp[b] / pb - rp[a] / pa;
      if (d) return d;
      d = mp[b] / pb - mp[a] / pa;
      if (d) return d;
      d = af[b] / pb - af[a] / pa;
      if (d) return d;
      d = tw[b] / pb - tw[a] / pa;
      if (d) return d;
      return a - b;
    });

    for (let rank0 = 0; rank0 < nTeams; rank0++) {
      const i = order[rank0];
      rankCounts[i][rank0] += 1;
      rankSum[i] += rank0 + 1;
      rpSum[i] += rp[i];
      rsSum[i] += played[i] ? rp[i] / played[i] : 0;
    }
  }

  const clampedCutoff = Math.max(1, Math.min(cutoff, nTeams));
  const currentOrder = {};
  rankOrder(base, teams).forEach((t, i) => (currentOrder[t] = i + 1));

  const remainingByTeam = {};
  for (const m of bundle.remaining) {
    for (const t of m.redCounting.concat(m.blueCounting)) {
      (remainingByTeam[t] = remainingByTeam[t] || []).push(m.key);
    }
  }

  const teamRows = teams.map((team, i) => {
    const counts = rankCounts[i];
    const dist = Array.from(counts, (c) => c / n);
    let best = nTeams;
    let worst = 1;
    for (let r = 0; r < nTeams; r++) {
      if (counts[r]) {
        best = Math.min(best, r + 1);
        worst = Math.max(worst, r + 1);
      }
    }
    return {
      team,
      currentRank: currentOrder[team] || nTeams,
      current: base[team],
      meanRank: rankSum[i] / n,
      medianRank: percentileFromCounts(counts, n, 0.5),
      p05Rank: percentileFromCounts(counts, n, 0.05),
      p95Rank: percentileFromCounts(counts, n, 0.95),
      bestRank: best,
      worstRank: worst,
      pRank1: dist[0],
      pTop3: dist.slice(0, 3).reduce((a, b) => a + b, 0),
      pCutoff: dist.slice(0, clampedCutoff).reduce((a, b) => a + b, 0),
      expectedRp: rpSum[i] / n,
      expectedRankScore: rsSum[i] / n,
      rankDist: dist,
      remainingMatches: remainingByTeam[team] || [],
    };
  });
  teamRows.sort((a, b) => a.meanRank - b.meanRank || b.pCutoff - a.pCutoff);

  const matchRows = bundle.remaining.map((m) => {
    const c = matchCounts[m.key];
    return {
      key: m.key,
      number: m.number,
      red: m.red,
      blue: m.blue,
      redSurrogates: m.redSurrogates,
      blueSurrogates: m.blueSurrogates,
      played: false,
      winner: "",
      pRed: c.red / n,
      pBlue: c.blue / n,
      pTie: c.tie / n,
      forced: forced[m.key] || "",
      redPredicted: m.red.reduce((a, t) => a + fit.hub[t], 0),
      bluePredicted: m.blue.reduce((a, t) => a + fit.hub[t], 0),
    };
  });

  return {
    teams: teamRows,
    matches: matchRows,
    meta: {
      n,
      seed,
      cutoff: clampedCutoff,
      oprUncertainty,
      useScouting,
      forced,
      forcedNudged: forcedNudges,
      remainingMatches: bundle.remaining.length,
    },
  };
}
