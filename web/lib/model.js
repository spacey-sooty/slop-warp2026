// Fit the performance model in the browser.
//
// Port of ranksim/model.py, kept deliberately line-for-line so the two can be
// diffed. It lives client-side because the scouting priors have to be
// recomputable whenever the refresh button pulls a newer scouting dump -- baking
// the fit at build time would freeze the priors and make the button cosmetic.
//
// tests/test_parity.py runs both implementations on the same bundle and asserts
// every rating, sigma, threshold and climb probability agrees to 1e-9 -- with
// the recency weights on and off, and at a half-life the bundle does not carry.

import { cholInvDiag, cholSolve, cholesky, mean, normalEquations } from "./linalg.js";

// Per-observation weight, halving every `halfLife` matches into the past, then
// rescaled to average 1 so ridge, sigma and the standard errors stay on the
// scale they would have without any decay. halfLife <= 0 turns it off.
export function recencyWeights(results, halfLife) {
  if (!results.length || !halfLife || halfLife <= 0) return results.map(() => 1);
  let latest = -Infinity;
  for (const r of results) if (r.matchNumber > latest) latest = r.matchNumber;
  const raw = results.map((r) => Math.pow(0.5, (latest - r.matchNumber) / halfLife));
  const total = raw.reduce((a, w) => a + w, 0);
  if (total <= 0) return results.map(() => 1);
  const scale = raw.length / total;
  return raw.map((w) => w * scale);
}

function weightedMean(values, weights) {
  let total = 0;
  for (const w of weights) total += w;
  if (total <= 0) return mean(values);
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += weights[i] * values[i];
  return sum / total;
}

function fitComponent(results, teams, index, values, ridge, prior, weights) {
  const rows = results.map((r) => r.teams.map((t) => index[t]));
  const w = weights || values.map(() => 1);
  const meanAlliance = weightedMean(values, w);
  const perTeamMean = meanAlliance / 3;

  const targets = {};
  for (const t of teams) {
    targets[t] = prior && prior[t] !== undefined ? prior[t] : perTeamMean;
  }
  const centered = values.map(
    (v, i) => v - results[i].teams.reduce((acc, t) => acc + targets[t], 0)
  );

  const { ata, aty } = normalEquations(rows, centered, teams.length, ridge, w);
  const lower = cholesky(ata);
  const deviations = cholSolve(lower, aty);

  const ratings = {};
  for (const t of teams) ratings[t] = targets[t] + deviations[index[t]];

  const residuals = values.map(
    (v, i) => v - results[i].teams.reduce((acc, t) => acc + ratings[t], 0)
  );
  const dof = Math.max(values.length - teams.length, 1);
  const sigma = Math.sqrt(residuals.reduce((a, e, i) => a + w[i] * e * e, 0) / dof);

  const totalSs = values.reduce((a, v, i) => a + w[i] * (v - meanAlliance) ** 2, 0);
  const residSs = residuals.reduce((a, e, i) => a + w[i] * e * e, 0);
  const rSquared = totalSs > 0 ? 1 - residSs / totalSs : 0;

  const invDiag = cholInvDiag(lower);
  const se = {};
  for (const t of teams) se[t] = sigma * Math.sqrt(Math.max(invDiag[index[t]], 0));

  return { ratings, se, sigma, rSquared };
}

// Least-squares line y = a + b*x over shared keys, plus its r^2. The r^2 becomes
// the weight the scouted prior carries, so scouting that does not predict this
// event's results quietly stops influencing the fit.
export function calibrate(x, y) {
  const keys = Object.keys(x).filter((k) => y[k] !== undefined);
  if (keys.length < 3) return { intercept: 0, slope: 0, rSquared: 0 };
  const xs = keys.map((k) => x[k]);
  const ys = keys.map((k) => y[k]);
  const mx = mean(xs);
  const my = mean(ys);
  const sxx = xs.reduce((a, v) => a + (v - mx) ** 2, 0);
  if (sxx <= 0) return { intercept: my, slope: 0, rSquared: 0 };
  let cov = 0;
  for (let i = 0; i < xs.length; i++) cov += (xs[i] - mx) * (ys[i] - my);
  const slope = cov / sxx;
  const intercept = my - slope * mx;
  let resid = 0;
  for (let i = 0; i < xs.length; i++) resid += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const total = ys.reduce((a, v) => a + (v - my) ** 2, 0);
  const rSquared = total > 0 ? 1 - resid / total : 0;
  return { intercept, slope, rSquared: Math.max(0, Math.min(1, rSquared)) };
}

function towerModel(results, teams, climbPrior, constants, weights) {
  const { autoTowerValue, endgameTowerValue, towerPriorWeight } = constants;
  const w = weights || results.map(() => 1);
  const perTeam = {};
  for (const t of teams) perTeam[t] = [];
  const pooled = [];

  results.forEach((res, i) => {
    res.teams.forEach((team, slot) => {
      const autoLevel = res.autoTowerLevels[slot];
      const endLevel = res.endgameTowerLevels[slot];
      const outcome = {
        autoLevel,
        endLevel,
        points: (autoTowerValue[autoLevel] || 0) + (endgameTowerValue[endLevel] || 0),
      };
      (perTeam[team] = perTeam[team] || []).push([outcome, w[i]]);
      pooled.push([outcome, w[i]]);
    });
  });

  const noClimb = { autoLevel: "None", endLevel: "None", points: 0 };
  const climbs = pooled.filter(([o]) => o.points > 0);
  // If nobody has climbed yet, a scouted climber is assumed to manage the
  // cheapest climb the game offers.
  const climbShapes = climbs.length
    ? climbs
    : [[{ autoLevel: "None", endLevel: "Level1", points: endgameTowerValue.Level1 }, 1]];

  // Split `mass` over a pool of outcomes in proportion to their weights.
  const spread = (pool, mass) => {
    const total = pool.reduce((a, [, weight]) => a + weight, 0);
    if (total <= 0) return [];
    return pool.map(([o, weight]) => [o, (mass * weight) / total]);
  };

  const priorFor = (team) => {
    if (climbPrior && climbPrior[team] !== undefined) {
      const p = Math.max(0, Math.min(1, climbPrior[team]));
      return spread(climbShapes, towerPriorWeight * p).concat([
        [noClimb, towerPriorWeight * (1 - p)],
      ]);
    }
    return spread(pooled, towerPriorWeight);
  };

  const distribution = (observations, prior) => {
    const counts = new Map();
    const bump = (o, weight) => {
      const key = `${o.autoLevel}|${o.endLevel}`;
      const row = counts.get(key) || [o, 0];
      row[1] += weight;
      counts.set(key, row);
    };
    for (const [o, weight] of observations) bump(o, weight);
    for (const [o, weight] of prior) bump(o, weight);
    let total = 0;
    for (const [, c] of counts.values()) total += c;
    if (total <= 0) total = 1;
    return [...counts.values()].map(([o, c]) => [o, c / total]);
  };

  const out = {};
  for (const t of teams) out[t] = distribution(perTeam[t] || [], priorFor(t));
  return out;
}

function learnThreshold(achieved, missed, name, constants) {
  const fallback = constants.defaultThresholds[name];
  if (achieved.length && missed.length) {
    const lo = Math.max(...missed);
    const hi = Math.min(...achieved);
    if (hi > lo) {
      return {
        value: (lo + hi) / 2,
        source: `learned from ${achieved.length} achieved / ${missed.length} not`,
      };
    }
    return { value: fallback, source: "overlapping observations, using 2026 default" };
  }
  if (achieved.length) {
    return {
      value: Math.min(...achieved),
      source: `lower bound from ${achieved.length} achieved`,
    };
  }
  return { value: fallback, source: "never achieved here, using 2026 default" };
}

function scoutedPrior(scouted, baseline, teams, weight, label) {
  const { intercept, slope, rSquared } = calibrate(scouted, baseline);
  const baseMean = mean(Object.values(baseline));
  const trust = Math.max(0, Math.min(1, weight * rSquared));
  const info = {
    metric: label,
    intercept,
    slope,
    rSquared,
    trust,
    teamsUsed: Object.keys(scouted).filter((t) => baseline[t] !== undefined).length,
  };
  if (trust <= 0 || slope <= 0) {
    info.applied = false;
    return { prior: null, info };
  }
  const prior = {};
  for (const team of teams) {
    if (scouted[team] !== undefined) {
      const predicted = Math.max(0, intercept + slope * scouted[team]);
      prior[team] = trust * predicted + (1 - trust) * baseMean;
    } else {
      prior[team] = baseMean;
    }
  }
  info.applied = true;
  return { prior, info };
}

// Scouted climb capability -> a per-match climb probability. Capability is not
// frequency, so it is calibrated against observed climb rates once the event has
// produced any, and flat-discounted until then.
function climbPriorFrom(results, scoutedTeams, weight, constants, weights) {
  const w = weights || results.map(() => 1);
  const observed = {};
  results.forEach((res, i) => {
    res.teams.forEach((team, slot) => {
      const climbed =
        res.autoTowerLevels[slot] !== "None" || res.endgameTowerLevels[slot] !== "None";
      (observed[team] = observed[team] || []).push([climbed, w[i]]);
    });
  });
  const rates = {};
  for (const [team, seen] of Object.entries(observed)) {
    const total = seen.reduce((a, [, rw]) => a + rw, 0);
    if (seen.length && total > 0) {
      rates[team] = seen.reduce((a, [c, rw]) => a + (c ? rw : 0), 0) / total;
    }
  }

  const scoutedRates = {};
  for (const [team, s] of Object.entries(scoutedTeams)) scoutedRates[team] = s.canClimbRate;

  const info = { metric: "canClimb report rate", teamsUsed: Object.keys(scoutedRates).length };
  const anyClimbs = Object.values(rates).some((r) => r > 0);
  if (anyClimbs) {
    const { intercept, slope, rSquared } = calibrate(scoutedRates, rates);
    Object.assign(info, { intercept, slope, rSquared });
    if (slope > 0) {
      info.mode = "calibrated against observed climbs";
      const prior = {};
      for (const [team, r] of Object.entries(scoutedRates)) {
        prior[team] = Math.max(
          constants.minClimbPrior,
          Math.min(0.95, intercept + slope * r)
        );
      }
      return { prior, info };
    }
  }
  info.mode = `no climbs observed yet, discounted by ${constants.defaultClimbTrust}`;
  const prior = {};
  for (const [team, r] of Object.entries(scoutedRates)) {
    prior[team] = Math.max(
      constants.minClimbPrior,
      Math.min(0.95, constants.defaultClimbTrust * weight * r)
    );
  }
  return { prior, info };
}

export function fit(
  bundle,
  { scouting = null, scoutingWeight = 1, ridge = null, halfLife = null } = {}
) {
  const constants = bundle.constants;
  const results = bundle.results;
  const teams = bundle.teams;
  const index = {};
  teams.forEach((t, i) => (index[t] = i));
  const lambda = ridge === null ? constants.ridge : ridge;
  const decay = halfLife === null ? constants.recencyHalfLife || 0 : halfLife;
  if (!results.length) throw new Error("no played qual matches to fit on");

  const weights = recencyWeights(results, decay);
  const hubValues = results.map((r) => r.hubPoints);
  const autoValues = results.map((r) => r.hubAuto);

  let hubFit = fitComponent(results, teams, index, hubValues, lambda, null, weights);
  let autoFit = fitComponent(results, teams, index, autoValues, lambda, null, weights);

  let scoutingInfo = { used: false };
  let climbPrior = null;

  if (scouting && scoutingWeight > 0) {
    const scoutedTeams = {};
    for (const [team, s] of Object.entries(scouting.teams)) {
      if (s.reports) scoutedTeams[team] = s;
    }
    const ballsBy = {};
    const autoBy = {};
    for (const [team, s] of Object.entries(scoutedTeams)) {
      ballsBy[team] = s.ballsPerMatch;
      autoBy[team] = s.autoBalls;
    }

    const hubPrior = scoutedPrior(
      ballsBy, hubFit.ratings, teams, scoutingWeight, "median balls per match"
    );
    const autoPrior = scoutedPrior(
      autoBy, autoFit.ratings, teams, scoutingWeight, "median auto balls"
    );
    if (hubPrior.prior) {
      hubFit = fitComponent(
        results, teams, index, hubValues, lambda, hubPrior.prior, weights
      );
    }
    if (autoPrior.prior) {
      autoFit = fitComponent(
        results, teams, index, autoValues, lambda, autoPrior.prior, weights
      );
    }
    const climb = climbPriorFrom(results, scoutedTeams, scoutingWeight, constants, weights);
    climbPrior = climb.prior;
    scoutingInfo = {
      used: true,
      reports: scouting.totalReports,
      teamsCovered: Object.keys(scoutedTeams).length,
      weight: scoutingWeight,
      hub: hubPrior.info,
      auto: autoPrior.info,
      climb: climb.info,
    };
  }

  const thresholds = {};
  const thresholdSources = {};
  const specs = [
    ["energized", (r) => r.energized, (r) => r.hubPoints],
    ["supercharged", (r) => r.supercharged, (r) => r.hubPoints],
    ["traversal", (r) => r.traversal, (r) => r.towerPoints],
  ];
  for (const [name, flag, value] of specs) {
    const achieved = results.filter(flag).map(value);
    const missed = results.filter((r) => !flag(r)).map(value);
    const learned = learnThreshold(achieved, missed, name, constants);
    thresholds[name] = learned.value;
    thresholdSources[name] = learned.source;
  }

  const towerOutcomes = towerModel(results, teams, climbPrior, constants, weights);
  const towerRate = (team) =>
    towerOutcomes[team].reduce((a, [o, p]) => a + (o.points > 0 ? p : 0), 0);
  const expectedTower = (team) =>
    towerOutcomes[team].reduce((a, [o, p]) => a + o.points * p, 0);

  // Effective sample size: how many equally-weighted matches the discounted
  // ones are worth, once the oldest count for a fraction of a match.
  let sumW = 0;
  let sumW2 = 0;
  for (const wi of weights) {
    sumW += wi;
    sumW2 += wi * wi;
  }
  const recency = {
    applied: decay > 0,
    halfLife: decay,
    effectiveObservations: sumW2 > 0 ? (sumW * sumW) / sumW2 : 0,
    oldestWeight: Math.min(...weights),
    newestWeight: Math.max(...weights),
  };

  return {
    teams,
    hub: hubFit.ratings,
    hubSe: hubFit.se,
    auto: autoFit.ratings,
    autoSe: autoFit.se,
    sigmaHub: hubFit.sigma,
    sigmaAuto: autoFit.sigma,
    rSquared: hubFit.rSquared,
    towerOutcomes,
    towerRate,
    expectedTower,
    foulSamples: results.map((r) => r.foulPoints),
    thresholds,
    thresholdSources,
    ridge: lambda,
    observations: results.length,
    meanAllianceHub: weightedMean(hubValues, weights),
    scouting: scoutingInfo,
    recency,
  };
}

// The shape /api/state used to return, so the rendering code is unchanged.
export function fitSummary(f) {
  const teams = {};
  for (const t of f.teams) {
    teams[t] = {
      hubOpr: f.hub[t],
      hubSe: f.hubSe[t],
      autoOpr: f.auto[t],
      towerRate: f.towerRate(t),
      expectedTower: f.expectedTower(t),
    };
  }
  return {
    sigmaHub: f.sigmaHub,
    sigmaAuto: f.sigmaAuto,
    ridge: f.ridge,
    observations: f.observations,
    meanAllianceHub: f.meanAllianceHub,
    rSquared: f.rSquared,
    scouting: f.scouting,
    recency: f.recency,
    thresholds: f.thresholds,
    thresholdSources: f.thresholdSources,
    teams,
  };
}
