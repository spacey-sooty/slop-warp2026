// The page is fully client-side: the baked event bundle carries everything TBA
// needs a key for, scouting is fetched live from the deployed app, and the fit
// and simulation both run here. No server, so no API to secure or pay for.

import { fit as fitModel, fitSummary } from "./lib/model.js";
import { build as buildScouting, fetchScouting } from "./lib/scouting.js";
import { simulate } from "./lib/simulate.js";
import {
  buildStandings,
  diffStandings,
  verifyAgainstPublished,
} from "./lib/standings.js";

const $ = (id) => document.getElementById(id);
const pct = (p, digits = 1) => `${(100 * p).toFixed(digits)}%`;
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

// Sequential blue ramp, binned. Six classes plus "never" keeps the scale
// readable and gives the legend discrete values to name.
const HEAT_BINS = [
  { max: 0.02, cssVar: "--seq-1", label: "≤2%" },
  { max: 0.05, cssVar: "--seq-2", label: "2–5%" },
  { max: 0.1, cssVar: "--seq-3", label: "5–10%" },
  { max: 0.25, cssVar: "--seq-4", label: "10–25%" },
  { max: 0.5, cssVar: "--seq-5", label: "25–50%" },
  { max: 1.01, cssVar: "--seq-6", label: ">50%" },
];

const state = {
  canRefreshTba: false, // true only when sim.py serve is behind the page
  bundle: null, // baked TBA data: standings, played results, remaining schedule
  event: null, // the shape the renderers read, mirroring the old /api/state
  fits: { scouted: null, plain: null }, // live fit objects the simulator needs
  scouting: null,
  result: null,
  // "bundle" = the standings the export shipped (the CSV, when there is one);
  // "rebuilt" = recomputed here from TBA's played matches.
  standingsMode: "bundle",
  rebuiltStandings: null,
  forced: {},
  sort: { key: "meanRank", dir: 1 },
  selected: null,
  running: false,
};

/* ------------------------------------------------------------------ boot */

async function boot() {
  wireControls();
  applyStoredTheme();
  try {
    const res = await fetch("data/event.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    state.bundle = await res.json();
  } catch (err) {
    showNotice(
      `Could not load the event bundle (data/event.json): ${err.message}. ` +
        "Regenerate it with ./sim.py export."
    );
    return;
  }

  $("cutoff").max = String(state.bundle.teams.length);
  applyBundleHalfLife();
  renderStandingsSource();
  await detectCapabilities();
  await loadScouting({ initial: true });
  await run();

  // ?team=9991 deep-links straight to a team's panel.
  const team = new URLSearchParams(location.search).get("team");
  if (team && state.bundle.teams.includes(team)) openDetail(team);
}

// A browser cannot call TBA -- that needs an API key, which must never ship to
// one -- so a live re-pull only exists when sim.py serve is behind the page. On a
// static host the button re-fetches the published bundle instead, picking up
// whatever the last export deployed. Probe once so the button can say which.
async function detectCapabilities() {
  try {
    const res = await fetch("api/capabilities", { cache: "no-store" });
    state.canRefreshTba = res.ok && (await res.json()).refreshTba === true;
  } catch {
    state.canRefreshTba = false;
  }
  $("tba-btn").title = state.canRefreshTba
    ? "Re-pull results and schedule from The Blue Alliance, then refit"
    : "Static host: re-checks for a newly published bundle. A live TBA pull needs ./sim.py serve, which holds the API key.";
}

// Pull the live scouting export and refit. Called on load and by the refresh
// button; a failure is never fatal, the page just falls back to the on-field fit.
async function loadScouting({ initial = false } = {}) {
  const bundle = state.bundle;
  let error = null;
  try {
    const raw = await fetchScouting(bundle.scoutingUrl, bundle.event.key);
    const dumpKey = (raw.event || {}).tbaKey;
    if (dumpKey && dumpKey !== bundle.event.key) {
      throw new Error(`export is for ${dumpKey}, not ${bundle.event.key}`);
    }
    state.scouting = buildScouting(raw, {
      teams: bundle.teams,
      aliases: bundle.csvAliases,
      tierOrder: bundle.constants.tierOrder,
    });
  } catch (err) {
    if (!initial) throw err;
    state.scouting = null;
    error = err.message;
  }

  refit(error);
  return error;
}

function refit(scoutingError = null) {
  const bundle = state.bundle;
  const weight = 1;
  const halfLife = Number($("half-life").value);
  state.fits.plain = fitModel(bundle, { scouting: null, halfLife });
  state.fits.scouted = state.scouting
    ? fitModel(bundle, { scouting: state.scouting, scoutingWeight: weight, halfLife })
    : state.fits.plain;
  // The same fit with the decay off, so the model table can show what recency
  // moved for each team the way it already does for scouting. A fit is ~8 ms.
  const flat = halfLife > 0
    ? fitModel(bundle, {
        scouting: state.scouting,
        scoutingWeight: weight,
        halfLife: 0,
      })
    : state.fits.scouted;
  const flatPlain =
    halfLife > 0 ? fitModel(bundle, { scouting: null, halfLife: 0 }) : state.fits.plain;

  state.event = {
    eventKey: bundle.event.key,
    eventName: bundle.event.name,
    teams: bundle.teams,
    matchesPlayed: bundle.event.matchesPlayed,
    matchesRemaining: bundle.event.matchesRemaining,
    generatedAt: bundle.generatedAt,
    csvAliases: bundle.csvAliases,
    csvDiscrepancies: bundle.csvDiscrepancies,
    tbaSource: bundle.tbaSource,
    tbaWarnings: bundle.tbaWarnings,
    standingsSource: bundle.standingsSource,
    fit: fitSummary(state.fits.scouted),
    fitPlain: fitSummary(state.fits.plain),
    fitFlat: fitSummary(flat),
    fitFlatPlain: fitSummary(flatPlain),
    scouting: state.scouting,
    scoutingError,
  };

  if (!state.scouting) {
    $("scouting").checked = false;
    $("scouting-field").classList.add("hidden");
  } else {
    $("scouting-field").classList.remove("hidden");
  }
  renderEventHeader();
  renderModelTable();
}

function wireControls() {
  $("run-btn").addEventListener("click", () => run());
  $("reset-btn").addEventListener("click", () => {
    state.forced = {};
    run();
  });
  $("theme-btn").addEventListener("click", toggleTheme);
  $("refresh-btn").addEventListener("click", refreshScouting);
  $("tba-btn").addEventListener("click", refreshTba);
  $("standings-btn").addEventListener("click", toggleStandings);
  $("scrim").addEventListener("click", closeDetail);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDetail();
  });
  for (const id of ["trials", "cutoff", "uncertainty", "scouting"]) {
    $(id).addEventListener("change", () => run());
  }
  // Recency changes the fit itself, not just how it is sampled, so this one
  // has to refit before it re-simulates.
  $("half-life").addEventListener("change", () => {
    refit(state.event ? state.event.scoutingError : null);
    run();
  });
}

// The bundle carries the half-life the CLI exported with, so the page starts on
// the same model a `./sim.py report` would print. An exported value that is not
// one of the offered steps gets an option of its own rather than being rounded.
function applyBundleHalfLife() {
  const select = $("half-life");
  const exported = String(state.bundle.constants.recencyHalfLife ?? 0);
  if (![...select.options].some((o) => o.value === exported)) {
    const option = el("option", null, `${exported} matches`);
    option.value = exported;
    select.append(option);
  }
  select.value = exported;
}

function applyStoredTheme() {
  // ?theme=light|dark pins the theme for a shared link or a pit-display kiosk.
  const fromUrl = new URLSearchParams(location.search).get("theme");
  const stored = fromUrl === "light" || fromUrl === "dark" ? fromUrl : localStorage.getItem("ranksim-theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);
}

function toggleTheme() {
  const current =
    document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("ranksim-theme", next);
  if (state.result) render();
}

function showNotice(text) {
  const node = $("notice");
  node.textContent = text;
  node.classList.remove("hidden");
}

/* ------------------------------------------------------------------ data */

async function run() {
  if (state.running) return;
  state.running = true;
  $("run-btn").disabled = true;
  document.querySelectorAll("section .card, section .match").forEach((c) => c.classList.add("running"));
  const seedRaw = $("seed").value.trim();
  const body = {
    n: Number($("trials").value),
    cutoff: Number($("cutoff").value) || 8,
    seed: seedRaw === "" ? null : Number(seedRaw),
    oprUncertainty: $("uncertainty").checked,
    useScouting: $("scouting").checked,
    forced: state.forced,
    standings: activeStandings(),
    standingsSource:
      state.standingsMode === "rebuilt" ? "tba-matches" : state.bundle.standingsSource,
  };
  try {
    // Yield a frame so the dimmed "running" state actually paints before the
    // sampler blocks the main thread. Time from after the frame, so the number
    // reported is the sampler's, not the paint's.
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    const started = performance.now();
    const fit = body.useScouting ? state.fits.scouted : state.fits.plain;
    state.result = simulate(state.bundle, fit, body);
    const ms = Math.round(performance.now() - started);
    const forcedCount = Object.keys(state.forced).length;
    $("status").textContent =
      `${body.n.toLocaleString()} trials in ${ms} ms` +
      (forcedCount ? ` · ${forcedCount} match${forcedCount > 1 ? "es" : ""} forced` : "");
    render();
  } catch (err) {
    showNotice(`Simulation failed: ${err.message}`);
  } finally {
    state.running = false;
    $("run-btn").disabled = false;
    document.querySelectorAll("section .card, section .match").forEach((c) => c.classList.remove("running"));
  }
}

// Re-pull the live scouting export, refit against it, and re-simulate. Forced
// what-ifs are deliberately kept: the scenario you were exploring survives a
// refresh, which is the whole point of pulling fresh data mid-event.
async function refreshTba() {
  const btn = $("tba-btn");
  const pill = $("tba-status");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  const before = state.bundle.generatedAt;
  try {
    let bundle = null;
    if (state.canRefreshTba) {
      const res = await fetch("api/refresh-tba", { method: "POST" });
      const payload = await res.json();
      if (payload.error) throw new Error(payload.error);
      bundle = payload;
    } else {
      // No key here; re-read the published bundle, bypassing the cache.
      const res = await fetch("data/event.json", { cache: "reload" });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      bundle = await res.json();
    }

    const played = bundle.event.matchesPlayed - state.bundle.event.matchesPlayed;
    state.bundle = bundle;
    // A newer pull means new matches to accumulate, so the rebuild has to be
    // redone rather than reused -- and its check against TBA's table with it.
    state.rebuiltStandings = null;
    // A forced what-if on a match that has since been played is meaningless.
    const stillRemaining = new Set(bundle.remaining.map((m) => m.key));
    for (const key of Object.keys(state.forced)) {
      if (!stillRemaining.has(key)) delete state.forced[key];
    }

    refit(state.event ? state.event.scoutingError : null);
    renderStandingsSource();

    // Say what actually happened. The client falls back to its disk cache when
    // TBA is unreachable, so "refreshed" is not the same as "reached TBA".
    const stale = bundle.tbaSource === "stale-cache";
    let label;
    if (played > 0) label = `+${played} match${played > 1 ? "es" : ""} played`;
    else if (stale) label = "TBA unreachable, cache";
    else if (!state.canRefreshTba && bundle.generatedAt <= before) {
      label = "no newer bundle published";
    } else label = "no new matches";
    pill.textContent = label;
    pill.classList.toggle("pill-warn", stale);
    pill.classList.remove("hidden");
    if (stale && (bundle.tbaWarnings || []).length) {
      showNotice(`TBA unreachable, showing cached results: ${bundle.tbaWarnings.join("; ")}`);
    }
    await run();
  } catch (err) {
    showNotice(`TBA refresh failed: ${err.message}. Showing the last good pull.`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh TBA";
  }
}

async function refreshScouting() {
  const btn = $("refresh-btn");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  const before = state.scouting ? state.scouting.totalReports : 0;
  try {
    await loadScouting();
    const after = state.scouting ? state.scouting.totalReports : 0;
    const delta = after - before;
    $("scouting-status").textContent =
      delta > 0
        ? `+${delta} new report${delta > 1 ? "s" : ""}`
        : "no new reports";
    $("scouting-status").classList.remove("hidden");
    await run();
  } catch (err) {
    showNotice(`Scouting refresh failed: ${err.message}. Showing the last good pull.`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh scouting";
  }
}

/* --------------------------------------------------------- standings source */

// Where the projection starts from. The bundle ships a snapshot of the current
// standings -- for this event a hand-supplied CSV, exported whenever someone
// last ran ./sim.py -- while TBA's match record is the live thing. Rebuilding
// from the played matches is a full re-derivation of the ranking rules here in
// the page (web/lib/standings.js), not a second table read off an endpoint,
// which is why it can then be checked against TBA's published one.
function rebuiltStandings() {
  if (!state.rebuiltStandings) {
    state.rebuiltStandings = buildStandings(state.bundle);
    state.standingsCheck = verifyAgainstPublished(
      state.rebuiltStandings,
      state.bundle.tbaRankings
    );
    state.standingsDiff = diffStandings(
      state.bundle.standings,
      state.rebuiltStandings,
      state.bundle.teams
    );
  }
  return state.rebuiltStandings;
}

function activeStandings() {
  return state.standingsMode === "rebuilt" ? rebuiltStandings() : state.bundle.standings;
}

function toggleStandings() {
  state.standingsMode = state.standingsMode === "rebuilt" ? "bundle" : "rebuilt";
  renderStandingsSource();
  if (state.event) renderEventHeader();
  run();
}

// Label the button by what pressing it does, and say what the switch changed.
function renderStandingsSource() {
  const btn = $("standings-btn");
  const pill = $("standings-status");
  const fromCsv = state.bundle.standingsSource === "csv";
  const exported = fromCsv ? "the CSV" : "the exported standings";
  const rebuilt = state.standingsMode === "rebuilt";
  // Computed either way: the tooltip should say what the button would do
  // before it is pressed, not only after.
  rebuiltStandings();
  const { changed, moved } = state.standingsDiff;
  const check = state.standingsCheck;

  btn.textContent = rebuilt
    ? `Back to ${fromCsv ? "CSV" : "exported"} standings`
    : "Use TBA standings";
  btn.title = rebuilt
    ? `Go back to the standings exported with the bundle (${exported})`
    : "Rebuild the current standings here from TBA's played matches, and start " +
      `the projection from those instead of ${exported}`;

  if (!rebuilt) {
    pill.classList.add("hidden");
    return;
  }
  const parts = [];
  parts.push(
    changed.length
      ? `${changed.length} team${changed.length > 1 ? "s" : ""} differ`
      : `identical to ${exported}`
  );
  if (moved.length) parts.push(`${moved.length} rank change${moved.length > 1 ? "s" : ""}`);
  if (check.checked) parts.push(check.agrees ? "matches TBA" : "differs from TBA");
  pill.textContent = parts.join(" · ");
  pill.classList.toggle("pill-warn", check.checked && !check.agrees);
  pill.classList.remove("hidden");
  pill.title = check.checked
    ? check.agrees
      ? "Rebuilt standings agree with TBA's published rankings, team for team"
      : `Disagrees with TBA's published rankings: ${check.mismatches.join("; ")}`
    : "TBA has published no rankings for this event to check against";

  if (check.checked && !check.agrees) {
    showNotice(
      "Standings rebuilt from the match breakdowns disagree with TBA's published " +
        `rankings: ${check.mismatches.slice(0, 4).join("; ")}` +
        (check.mismatches.length > 4 ? ` (+${check.mismatches.length - 4} more)` : "")
    );
  }
}

function relativeTime(ms) {
  if (!ms) return "unknown";
  const mins = Math.round((Date.now() - ms) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/* ------------------------------------------------------------ rendering */

function renderEventHeader() {
  const ev = state.event;
  $("event-name").textContent = ev.eventName;
  const parts = [
    `${ev.eventKey} · ${ev.matchesPlayed} qualification matches played, ` +
      `${ev.matchesRemaining} remaining · ${ev.teams.length} teams`,
    `results ${relativeTime(ev.generatedAt * 1000)}` +
      (ev.tbaSource === "stale-cache" ? " (cached, TBA unreachable)" : ""),
  ];
  if (ev.scouting) {
    parts.push(
      `scouting ${relativeTime(ev.scouting.fetchedAt)} ` +
        `(${ev.scouting.totalReports} reports)`
    );
  }
  parts.push(
    state.standingsMode === "rebuilt"
      ? "standings rebuilt from TBA matches"
      : `standings from ${ev.standingsSource === "csv" ? "the CSV" : "TBA matches"}`
  );
  $("event-sub").textContent = parts.join(" · ");
  const notes = [];
  if (Object.keys(ev.csvAliases || {}).length) {
    notes.push(
      Object.entries(ev.csvAliases)
        .map(([csv, tba]) => `CSV team ${csv} matched to TBA team ${tba}`)
        .join("; ")
    );
  }
  if ((ev.csvDiscrepancies || []).length) {
    notes.push(`CSV/TBA differences: ${ev.csvDiscrepancies.join("; ")}`);
  }
  if (ev.scoutingError) notes.push(`Scouting unavailable: ${ev.scoutingError}`);
  for (const note of (ev.scouting || {}).unmatched || []) {
    notes.push(`Scouted team not at this event, ignored: ${note}`);
  }
  if (notes.length) showNotice(notes.join(" · "));
}

function render() {
  renderTiles();
  renderStandings();
  renderHeatmap();
  renderMatches();
  renderModelTable();
  if (state.selected) openDetail(state.selected);
  $("reset-btn").classList.toggle("hidden", Object.keys(state.forced).length === 0);
}

function activeFit() {
  const useScouting = state.result ? state.result.meta.useScouting : true;
  return useScouting ? state.event.fit : state.event.fitPlain;
}

function renderTiles() {
  const { teams, meta } = state.result;
  const leader = teams.reduce((a, b) => (a.pRank1 >= b.pRank1 ? a : b));
  const bubble = teams.filter((t) => t.pCutoff > 0.1 && t.pCutoff < 0.9);
  const locked = teams.filter((t) => t.pCutoff >= 0.999).length;
  const fit = activeFit();

  const tiles = [
    {
      label: "Matches remaining",
      value: meta.remainingMatches,
      note: `${state.event.matchesPlayed} played`,
    },
    {
      label: "Most likely #1 seed",
      value: leader.team,
      note: `${pct(leader.pRank1)} of trials`,
    },
    {
      label: `On the bubble for top ${meta.cutoff}`,
      value: bubble.length,
      note: locked ? `${locked} already locked in` : "no team is mathematically safe",
    },
    {
      label: "Model fit",
      value: `R² ${fit.rSquared.toFixed(2)}`,
      note: `±${fit.sigmaHub.toFixed(0)} pts per alliance, ${fit.observations} appearances`,
    },
  ];

  const scouting = state.event.scouting;
  if (scouting) {
    tiles.push(
      fit.scouting && fit.scouting.used
        ? {
            label: "Pit scouting",
            value: scouting.totalReports,
            note:
              `reports over ${scouting.teamsCovered} teams · ` +
              `trust ${fit.scouting.hub.trust.toFixed(2)}`,
          }
        : {
            label: "Pit scouting",
            value: "off",
            note: `${scouting.totalReports} reports available`,
          }
    );
  }

  const wrap = $("tiles");
  wrap.replaceChildren();
  for (const t of tiles) {
    const node = el("div", "tile");
    node.append(el("div", "label", t.label), el("div", "value", String(t.value)), el("div", "note", t.note));
    wrap.append(node);
  }
}

const COLUMNS = [
  { key: "currentRank", label: "Now", title: "Current rank", fmt: (r) => r.currentRank },
  { key: "team", label: "Team", cls: "left", fmt: (r) => r.team, cell: "team-cell" },
  { key: "rp", label: "RP", fmt: (r) => r.current.rp, sortValue: (r) => -r.current.rp },
  {
    key: "rankScore",
    label: "RS",
    title: "Ranking score, RP per match",
    fmt: (r) => r.current.rankScore.toFixed(2),
    sortValue: (r) => -r.current.rankScore,
  },
  { key: "meanRank", label: "Proj. rank", fmt: (r) => r.meanRank.toFixed(1) },
  { key: "range", label: "5th–95th pct", sortable: false, kind: "range" },
  { key: "pRank1", label: "P(#1)", fmt: (r) => pct(r.pRank1), sortValue: (r) => -r.pRank1 },
  { key: "pCutoff", label: "P(top N)", kind: "prob", sortValue: (r) => -r.pCutoff },
  { key: "expectedRp", label: "E[RP]", fmt: (r) => r.expectedRp.toFixed(1), sortValue: (r) => -r.expectedRp },
  { key: "delta", label: "Δ", title: "Projected rank minus current rank", kind: "delta", sortValue: (r) => r.meanRank - r.currentRank },
];

function sortedTeams() {
  const rows = [...state.result.teams];
  const col = COLUMNS.find((c) => c.key === state.sort.key);
  const value = col && col.sortValue ? col.sortValue : (r) => r[state.sort.key];
  rows.sort((a, b) => {
    const av = value(a);
    const bv = value(b);
    if (av === bv) return a.meanRank - b.meanRank;
    return (av > bv ? 1 : -1) * state.sort.dir;
  });
  return rows;
}

function renderStandings() {
  const nTeams = state.event.teams.length;
  const cutoff = state.result.meta.cutoff;

  const head = $("standings-head");
  head.replaceChildren();
  for (const col of COLUMNS) {
    const th = el("th", col.cls || "");
    th.textContent = col.label === "P(top N)" ? `P(top ${cutoff})` : col.label;
    if (col.title) th.title = col.title;
    if (col.sortable === false) {
      th.style.cursor = "default";
    } else {
      th.addEventListener("click", () => {
        if (state.sort.key === col.key) state.sort.dir *= -1;
        else state.sort = { key: col.key, dir: 1 };
        renderStandings();
      });
      if (state.sort.key === col.key) {
        th.append(el("span", "arrow", state.sort.dir === 1 ? " ↑" : " ↓"));
      }
    }
    head.append(th);
  }

  const body = $("standings-body");
  body.replaceChildren();
  for (const row of sortedTeams()) {
    const tr = el("tr");
    if (state.selected === row.team) tr.classList.add("selected");
    tr.addEventListener("click", () => openDetail(row.team));
    for (const col of COLUMNS) {
      const td = el("td", [col.cls, col.cell].filter(Boolean).join(" "));
      if (col.kind === "range") {
        td.classList.add("range-cell");
        td.append(rangeBar(row, nTeams));
        td.title = `5th percentile rank ${row.p05Rank}, median ${row.medianRank}, 95th percentile ${row.p95Rank}`;
      } else if (col.kind === "prob") {
        td.append(probCell(row.pCutoff));
      } else if (col.kind === "delta") {
        const delta = row.currentRank - row.meanRank;
        const flat = Math.abs(delta) < 0.05;
        td.textContent = flat ? "0.0" : delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1);
        td.className += flat ? " muted" : delta > 0 ? " delta-up" : " delta-down";
      } else {
        td.textContent = col.fmt(row);
      }
      tr.append(td);
    }
    body.append(tr);
  }
}

function rangeBar(row, nTeams) {
  const wrap = el("div", "rangebar");
  const scale = (rank) => ((rank - 1) / Math.max(nTeams - 1, 1)) * 100;
  const track = el("div", "track");
  const span = el("div", "span");
  span.style.left = `${scale(row.p05Rank)}%`;
  span.style.width = `${Math.max(scale(row.p95Rank) - scale(row.p05Rank), 1.5)}%`;
  const median = el("div", "median");
  median.style.left = `${scale(row.medianRank)}%`;
  wrap.append(track, span, median);
  return wrap;
}

function probCell(p) {
  const wrap = el("div", "probcell");
  const bar = el("div", "probbar");
  const fill = el("i");
  fill.style.width = `${Math.max(p * 100, p > 0 ? 2 : 0)}%`;
  bar.append(fill);
  wrap.append(el("span", "", pct(p)), bar);
  return wrap;
}

/* ------------------------------------------------------------- heatmap */

function heatColor(p) {
  if (p <= 0) return null;
  for (const bin of HEAT_BINS) if (p <= bin.max) return `var(${bin.cssVar})`;
  return `var(${HEAT_BINS[HEAT_BINS.length - 1].cssVar})`;
}

function renderHeatmap() {
  const rows = sortedTeams();
  const nTeams = state.event.teams.length;

  const legend = $("heat-legend");
  legend.replaceChildren();
  legend.append(el("span", "", "Chance of finishing at rank:"));
  const never = el("span", "swatch");
  const neverSwatch = el("i");
  neverSwatch.style.background = "var(--surface-sunken)";
  never.append(neverSwatch, el("span", "", "never"));
  legend.append(never);
  for (const bin of HEAT_BINS) {
    const sw = el("span", "swatch");
    const chip = el("i");
    chip.style.background = `var(${bin.cssVar})`;
    sw.append(chip, el("span", "", bin.label));
    legend.append(sw);
  }

  const grid = $("heat-grid");
  grid.replaceChildren();
  grid.style.gridTemplateColumns = `56px repeat(${nTeams}, minmax(15px, 1fr))`;

  for (const row of rows) {
    const label = el("div", "heat-label", row.team);
    grid.append(label);
    for (let rank = 1; rank <= nTeams; rank++) {
      const p = row.rankDist[rank - 1] || 0;
      const cell = el("div", "heat-cell");
      const color = heatColor(p);
      if (color) cell.style.background = color;
      else cell.classList.add("empty");
      cell.addEventListener("mouseenter", (e) =>
        showTooltip(e, `${row.team} · rank ${rank} · ${p > 0 ? pct(p) : "never"}`)
      );
      cell.addEventListener("mousemove", moveTooltip);
      cell.addEventListener("mouseleave", hideTooltip);
      grid.append(cell);
    }
  }

  grid.append(el("div", ""));
  for (let rank = 1; rank <= nTeams; rank++) {
    grid.append(el("div", "heat-axis", rank % 2 === 1 || nTeams <= 14 ? String(rank) : ""));
  }
}

/* -------------------------------------------------------------- matches */

function renderMatches() {
  const list = $("matchlist");
  list.replaceChildren();
  const teamRank = new Map(state.result.teams.map((t) => [t.team, t.currentRank]));

  for (const m of state.result.matches) {
    const row = el("div", "match" + (m.forced ? " forced" : ""));
    row.append(el("div", "num", `Q${m.number}`));

    const teams = el("div", "teams-row");
    teams.append(allianceChip("red", m.red, m.redSurrogates, teamRank));
    teams.append(el("span", "vs", "vs"));
    teams.append(allianceChip("blue", m.blue, m.blueSurrogates, teamRank));
    row.append(teams);

    const probWrap = el("div", "probwrap");
    const bar = el("div", "splitbar");
    const red = el("i");
    red.style.width = `${m.pRed * 100}%`;
    const tie = el("i", "tie");
    tie.style.width = `${m.pTie * 100}%`;
    const blue = el("i");
    blue.style.width = `${m.pBlue * 100}%`;
    bar.append(red, tie, blue);
    bar.title = `red ${pct(m.pRed)} · tie ${pct(m.pTie)} · blue ${pct(m.pBlue)}`;

    const labels = el("div", "splitlabels");
    labels.append(el("span", "", `Red ${pct(m.pRed, 0)}`));
    if (m.pTie >= 0.005) labels.append(el("span", "muted", `tie ${pct(m.pTie, 0)}`));
    labels.append(el("span", "", `Blue ${pct(m.pBlue, 0)}`));

    const btns = el("div", "forcebtns");
    for (const [value, text] of [["", "Auto"], ["red", "Red"], ["tie", "Tie"], ["blue", "Blue"]]) {
      const b = el("button", "", text);
      b.setAttribute("aria-pressed", String((m.forced || "") === value));
      b.addEventListener("click", () => {
        if (value === "") delete state.forced[m.key];
        else state.forced[m.key] = value;
        run();
      });
      btns.append(b);
    }

    probWrap.append(bar, labels, btns);
    row.append(probWrap);
    list.append(row);
  }
}

function allianceChip(color, teams, surrogates, teamRank) {
  const wrap = el("div", `alliance ${color}`);
  wrap.append(el("span", "dot"));
  const names = el("span", "names");
  teams.forEach((t, i) => {
    if (i) names.append(document.createTextNode(" "));
    const b = el("b", "", t);
    b.style.cursor = "pointer";
    b.title = `Team ${t}, currently rank ${teamRank.get(t) ?? "?"}`;
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      openDetail(t);
    });
    names.append(b);
    if (surrogates.includes(t)) names.append(el("span", "muted", "*"));
  });
  wrap.append(names);
  return wrap;
}

/* --------------------------------------------------------------- detail */

function openDetail(team) {
  const row = state.result.teams.find((t) => t.team === team);
  if (!row) return;
  state.selected = team;
  const panel = $("detail");
  panel.replaceChildren();
  panel.setAttribute("aria-hidden", "false");

  const close = el("button", "close", "×");
  close.setAttribute("aria-label", "Close");
  close.addEventListener("click", closeDetail);
  panel.append(close);

  panel.append(el("h3", "", `Team ${team}`));
  const cur = row.current;
  panel.append(
    el(
      "p",
      "subtitle",
      `Currently rank ${row.currentRank} · ${cur.wins}-${cur.losses}-${cur.ties} · ` +
        `${cur.rp} RP in ${cur.played} matches`
    )
  );

  const stats = el("div", "statgrid");
  const cutoff = state.result.meta.cutoff;
  for (const [k, v] of [
    ["Projected rank", row.meanRank.toFixed(1)],
    ["Median rank", String(row.medianRank)],
    [`P(top ${cutoff})`, pct(row.pCutoff)],
    ["P(#1 seed)", pct(row.pRank1)],
    ["Expected RP", row.expectedRp.toFixed(1)],
    ["Expected RS", row.expectedRankScore.toFixed(2)],
  ]) {
    const box = el("div");
    box.append(el("div", "k", k), el("div", "v", v));
    stats.append(box);
  }
  panel.append(stats);

  panel.append(el("h4", "", "Final rank distribution"));
  const nTeams = state.event.teams.length;
  const peak = Math.max(...row.rankDist);
  const hist = el("div", "hist");
  row.rankDist.forEach((p, i) => {
    const bar = el("div", "bar");
    // Ranks the team never reaches get no mark at all -- a 1px stub on every
    // column reads as a dashed axis rule.
    bar.style.height = p > 0 && peak > 0 ? `${Math.max((p / peak) * 100, 1.5)}%` : "0";
    bar.title = `rank ${i + 1} · ${p > 0 ? pct(p) : "never"}`;
    hist.append(bar);
  });
  panel.append(hist);
  const axis = el("div", "hist-axis");
  for (let r = 1; r <= nTeams; r++) axis.append(el("span", "", r % 5 === 0 || r === 1 ? String(r) : ""));
  panel.append(axis);

  panel.append(el("h4", "", `Remaining matches (${row.remainingMatches.length})`));
  const table = el("table");
  const tbody = el("tbody");
  for (const key of row.remainingMatches) {
    const m = state.result.matches.find((x) => x.key === key);
    if (!m) continue;
    const onRed = m.red.includes(team);
    const win = onRed ? m.pRed : m.pBlue;
    const tr = el("tr");
    tr.append(el("td", "left", `Q${m.number}`));
    tr.append(el("td", "left", onRed ? "Red" : "Blue"));
    const partners = (onRed ? m.red : m.blue).filter((t) => t !== team).join(", ");
    const against = (onRed ? m.blue : m.red).join(", ");
    tr.append(el("td", "left muted", `with ${partners} vs ${against}`));
    const p = el("td", "", pct(win, 0) + (m.forced ? " ⃰" : ""));
    if (m.forced) p.title = `Outcome forced to ${m.forced}`;
    tr.append(p);
    tbody.append(tr);
  }
  table.append(tbody);
  panel.append(table);

  const scout = state.event.scouting ? state.event.scouting.teams[team] : null;
  if (scout && scout.reports) {
    panel.append(el("h4", "", `Pit scouting (${scout.reports} reports)`));
    const st = el("table");
    const sb = el("tbody");
    const rows = [
      ["Balls per match", `${scout.ballsPerMatch.toFixed(0)} median`],
      ["Auto balls", scout.hasAutoRate > 0 ? `${scout.autoBalls.toFixed(0)} median` : "no auto reported"],
      ["Can climb", scout.canClimbRate > 0 ? `${pct(scout.canClimbRate, 0)} of reports` : "not reported"],
      ["Driver rating", `${scout.driverRating.toFixed(1)} / 6`],
      ["Defense rating", `${scout.defenseRating.toFixed(1)} / 6`],
      ["Storage capacity", scout.storage.toFixed(0)],
    ];
    if (scout.picklistRank !== null) {
      rows.push([
        "Picklist consensus",
        `#${scout.picklistRank.toFixed(1)} avg` +
          (scout.picklistTier ? ` · ${scout.picklistTier} tier` : "") +
          ` · ${scout.picklistLists} lists`,
      ]);
    }
    if (scout.primaryTier) {
      rows.push(["Primary list", `#${scout.primaryRank} · ${scout.primaryTier} tier`]);
    }
    for (const [k, v] of rows) {
      const tr = el("tr");
      tr.append(el("td", "left muted", k), el("td", "", String(v)));
      sb.append(tr);
    }
    st.append(sb);
    panel.append(st);

    if (scout.tags.length) {
      const tagWrap = el("div", "tags");
      for (const [tag, count] of scout.tags) {
        tagWrap.append(el("span", "tag", count > 1 ? `${tag} ×${count}` : tag));
      }
      panel.append(tagWrap);
    }
    if (scout.notes.length) {
      const notes = el("div", "scout-notes");
      for (const note of scout.notes) notes.append(el("p", "", note));
      panel.append(notes);
    }
  }

  const model = activeFit().teams[team];
  if (model) {
    panel.append(el("h4", "", "Fitted ratings"));
    const mt = el("table");
    const mb = el("tbody");
    for (const [k, v] of [
      ["Hub OPR", `${model.hubOpr.toFixed(1)} ± ${model.hubSe.toFixed(1)}`],
      ["Auto fuel OPR", model.autoOpr.toFixed(1)],
      ["Climb rate", pct(model.towerRate, 0)],
      ["Expected tower pts", model.expectedTower.toFixed(1)],
    ]) {
      const tr = el("tr");
      tr.append(el("td", "left muted", k), el("td", "", v));
      mb.append(tr);
    }
    mt.append(mb);
    panel.append(mt);
  }

  panel.classList.add("open");
  $("scrim").classList.add("open");
  renderStandings();
}

function closeDetail() {
  state.selected = null;
  $("detail").classList.remove("open");
  $("detail").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
  if (state.result) renderStandings();
}

/* -------------------------------------------------------------- model */

function renderModelTable() {
  const fit = activeFit();
  const useScouting = state.result ? state.result.meta.useScouting : true;
  const plain = state.event.fitPlain;
  // Same model as the active one, minus the recency decay -- so the Δ column
  // isolates recency rather than mixing it with the scouting toggle.
  const flat = useScouting ? state.event.fitFlat : state.event.fitFlatPlain;
  const scouting = state.event.scouting;

  const rec = fit.recency;
  const appearances = rec && rec.applied
    ? `${fit.observations} alliance appearances, ` +
      `discounted by recency to ${rec.effectiveObservations.toFixed(1)} effective ` +
      `(half-life ${rec.halfLife} matches: the oldest counts ${rec.oldestWeight.toFixed(2)} ` +
      `against the newest at ${rec.newestWeight.toFixed(2)})`
    : `${fit.observations} alliance appearances, every match weighted equally`;

  let summary =
    `Ridge-regularised OPR on ${appearances} · ` +
    `R² ${fit.rSquared.toFixed(3)} · residual σ ${fit.sigmaHub.toFixed(1)} hub points · ` +
    `RP thresholds: energized ≥ ${fit.thresholds.energized} hub, ` +
    `supercharged ≥ ${fit.thresholds.supercharged} hub, ` +
    `traversal ≥ ${fit.thresholds.traversal} tower`;
  const sc = fit.scouting;
  if (sc && sc.used) {
    summary +=
      ` · scouting priors from ${sc.reports} pit reports: ` +
      `${sc.hub.metric} explains ${pct(sc.hub.rSquared, 0)} of hub OPR (trust ${sc.hub.trust.toFixed(2)}), ` +
      `${sc.auto.metric} ${pct(sc.auto.rSquared, 0)} of auto (trust ${sc.auto.trust.toFixed(2)}); ` +
      `climb prior ${sc.climb.mode}`;
  } else if (scouting) {
    summary += " · scouting priors off";
  }
  $("model-summary").textContent = summary;

  const body = $("model-body");
  body.replaceChildren();
  const entries = Object.entries(fit.teams).sort((a, b) => b[1].hubOpr - a[1].hubOpr);
  for (const [team, m] of entries) {
    const tr = el("tr");
    tr.addEventListener("click", () => state.result && openDetail(team));
    tr.append(el("td", "left team-cell", team));
    tr.append(el("td", "", m.hubOpr.toFixed(1)));
    tr.append(el("td", "muted", m.hubSe.toFixed(1)));

    const deltaCell = (from) => {
      const base = from.teams[team] ? from.teams[team].hubOpr : m.hubOpr;
      const delta = m.hubOpr - base;
      const cell = el("td", Math.abs(delta) < 0.05 ? "muted" : "");
      cell.textContent =
        Math.abs(delta) < 0.05 ? "—" : (delta > 0 ? "+" : "") + delta.toFixed(1);
      return cell;
    };
    tr.append(deltaCell(plain));
    tr.append(deltaCell(flat));

    const st = scouting ? scouting.teams[team] : null;
    const scell = el("td", st && st.reports ? "" : "muted");
    scell.textContent = st && st.reports ? st.ballsPerMatch.toFixed(0) : "—";
    if (st && st.reports) scell.title = `median of ${st.reports} pit reports`;
    tr.append(scell);

    tr.append(el("td", "", m.autoOpr.toFixed(1)));
    tr.append(el("td", "", pct(m.towerRate, 0)));
    tr.append(el("td", "", m.expectedTower.toFixed(1)));
    body.append(tr);
  }
}

/* ------------------------------------------------------------- tooltip */

function showTooltip(event, text) {
  const tip = $("tooltip");
  tip.textContent = text;
  tip.classList.add("show");
  moveTooltip(event);
}

function moveTooltip(event) {
  const tip = $("tooltip");
  const pad = 14;
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  const rect = tip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTooltip() {
  $("tooltip").classList.remove("show");
}

boot();
