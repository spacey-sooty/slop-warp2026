# End-of-event ranking simulator

Projects final qualification rankings for an FRC event by Monte Carlo: it takes
the current standings, fits a performance model to the matches already played,
plays out every remaining match a few thousand times, and reports where each team
is likely to finish.

Built for **2026auwarp** (West Australian Robotics Playoffs) using
`../event-rankings.csv` for the current standings, The Blue Alliance for the
schedule, results and score breakdowns, and **scoutinapp** for pit scouting.
Pure Python standard library — no dependencies, no build step.

```
./sim.py serve            # export + serve the site on http://127.0.0.1:8765
./sim.py export           # rebuild web/data/event.json only
./sim.py report           # text projection
./sim.py fit              # fitted team ratings and RP thresholds
./sim.py scouting         # what the scouting export contributed
./sim.py backtest         # walk-forward check, with and without scouting
./sim.py fetch            # force a fresh pull from TBA and scoutinapp
```

## The site is static

`web/` is a self-contained static site — no server, no build step, no
dependencies. It loads a baked `data/event.json`, **fetches scouting live from
the deployed scoutinapp**, and runs both the model fit and the Monte Carlo in
the browser.

| | where it comes from | how it refreshes |
|---|---|---|
| Played results, schedule, standings | baked into `data/event.json` (TBA needs a key, which must never ship to a browser) | **Refresh TBA** button |
| Pit scouting | fetched at page load from the deployed app | **Refresh scouting** button |
| Current standings | the exported snapshot, or [rebuilt in the page](#standings-the-csv-or-the-matches) from the played matches | **Use TBA standings** button |

The split falls out of the key: TBA's API needs one, so its data is resolved
somewhere trusted; scoutinapp's export needs none and serves
`Access-Control-Allow-Origin: *`, so the page reads it directly.

**Refresh scouting** always does the real thing: re-pull the export, **refit the
model against it**, re-simulate.

**Refresh TBA** does what it can, and says which:

| running under | what it does |
|---|---|
| `./sim.py serve` | posts to `/api/refresh-tba`; the server holds the key, re-pulls TBA, rewrites the bundle and returns it — a true live pull |
| a static host | re-fetches `data/event.json` past the cache, picking up whatever the last `./sim.py export` published |

The page probes `/api/capabilities` on load, so the button's tooltip describes
the mode it is actually in. Both refreshes keep any forced what-if scenario,
except on matches that have since been played — those are dropped, since forcing
a played match is meaningless.

> TBA cannot be fetched from the browser directly. It answers `200` to a
> keyless request from curl, which looks like it could be — but that is a
> Cloudflare edge-cache hit. With an `Origin` header the request goes to the
> origin and returns `401`, so a browser gets nothing without a key.

That is also why the fit is reimplemented in JavaScript rather than baked: if the
priors were frozen at export time, the refresh button would be cosmetic.

### Deploying

**GitHub Pages** (`.github/workflows/deploy.yml`) is set up. It regenerates the
bundle from a live TBA pull, gates on the test suite, and publishes `web/`. Two
one-time steps before the first run:

1. **Settings → Secrets and variables → Actions →** add `TBA_API_KEY`
   (from [thebluealliance.com/account](https://www.thebluealliance.com/account)).
2. **Settings → Pages → Source: GitHub Actions.**

Then push, or run it from the Actions tab. It also runs every 15 minutes so
results stay current during an event — drop the `schedule:` block once the event
is over.

The workflow **fails rather than publishing stale data**: if the TBA pull falls
back to cache, or the bundle is more than 10 minutes old, the deploy stops. A
site that quietly shows yesterday's standings is worse than one that visibly
failed to update.

Any other static host works too — point it at `web/`. A `vercel.json` is
included for `npx vercel deploy --prod`.

Relative URLs throughout, so serving from a subpath
(`user.github.io/repo/`) works without configuration.

### Two implementations, one model

`ranksim/model.py` and `web/lib/model.js` are the same statistics twice, which is
a genuine maintenance hazard. `tests/test_parity.py` is the guard: it runs both
over the same bundle and asserts every rating, standard error, sigma, threshold,
calibration coefficient, recency weight and climb probability agrees to
**1e-9**, plus the whole scouting reduction (medians, rates, tags, picklist
consensus) and the standings rebuilt in the browser (every field of every team's
record, and the ranking order) exactly. The recency check runs at three half-lives, one of them not
the bundle's, so neither side can pass by falling back to the same default. Change one
without the other and the suite fails.

The samplers use different RNGs, so a seed does not reproduce across them; that
one is checked statistically instead.

### A refresh that cannot lie

`TBAClient` falls back to its disk cache when TBA is unreachable, which is the
right behaviour on venue wifi — but it used to do so *silently*, so a refresh
that reached nothing looked identical to one that reached TBA. Two bugs came out
of that:

- TBA answers **403 to urllib's default `Python-urllib/3.13` User-Agent**, so
  every live fetch had been failing and quietly serving cache. Fixed by sending a
  real User-Agent.
- The fallback is now recorded (`tbaSource`, `tbaWarnings`) and surfaced: the
  pill reads "TBA unreachable, cache" in red and the header appends "(cached,
  TBA unreachable)".

Both are covered in `tests/test_ranking.py`.

Performance moved the right way. Measured in Chrome against the old Python
server's 14,300 trials/sec:

| trials | browser | |
|---|---|---|
| 5,000 | 77 ms | the default |
| 50,000 | 469 ms | |
| 200,000 | 1.8 s | ~112,000 trials/sec at steady state |

The fit itself is 8 ms, which is why **Refresh scouting** can refit and
re-simulate without a visible pause.

## Web UI

`./sim.py serve` gives you:

- **Projected final standings** — projected rank, 5th–95th percentile band,
  P(#1 seed), P(top 8), expected RP. Sortable; click a team for its full rank
  distribution, remaining schedule and fitted ratings.
- **Rank probability heatmap** — chance of every team finishing at every rank.
- **Remaining matches** — win probability per match, with **Red / Tie / Blue**
  buttons that force an outcome and re-run instantly. That is the what-if tool:
  force a team's matches and watch their playoff odds move.
- **Model** — the fitted ratings and the RP thresholds the simulation is using.
- **Use TBA standings** — recompute the current standings from the played
  matches instead of the exported snapshot, and start the projection from those.
  See [Standings: the CSV, or the matches](#standings-the-csv-or-the-matches).

Controls at the top set trial count, the cutoff rank (default 8, the alliance
captain line), a fixed seed for reproducibility, the [recency
half-life](#recency), whether to include rating uncertainty, and whether to fold
in pit scouting. Changing the half-life refits in the page and re-runs, so the
effect of trusting recent matches more is visible immediately. `?theme=light` / `?theme=dark`
pins the theme for a pit display; `?team=9991` deep-links to a team's panel.

## Ranking rules (2026)

Reverse-engineered from TBA score breakdowns and verified against six 2026 events
— `tests/test_ranking.py` checks the rebuilt standings reproduce TBA's published
rankings exactly, for every team and every tiebreaker.

| | |
|---|---|
| Ranking points | 3 win / 1 tie / 0 loss, plus 1 RP each for **Energized**, **Supercharged** and **Traversal** |
| Energized | alliance hub points ≥ threshold (100 at regionals; **75** at this event) |
| Supercharged | alliance hub points ≥ threshold (~360 at regionals; ~213 here) |
| Traversal | alliance tower points ≥ 50 |
| Match points | `totalPoints − foulPoints` — points awarded from opponent fouls decide the *match* but not the *tiebreaker* |
| Avg Auto Fuel | `hubScore.autoPoints` |
| Avg Tower | `totalTowerPoints` |
| Sort | Ranking Score (RP ÷ played), then Avg Match, Avg Auto Fuel, Avg Tower |
| Surrogates | score for the alliance, do not count toward the surrogate's own ranking |

Thresholds are **learned from the event's own played matches** (the midpoint of
the achieved / not-achieved boundary) rather than hardcoded, so this works at an
offseason event running modified thresholds — as 2026auwarp does. Regional values
are only a fallback for an achievement nobody has managed yet.

### Standings: the CSV, or the matches

A projection starts from the current standings, and there are two of them. The
bundle ships a snapshot — for this event the hand-supplied `event-rankings.csv`,
which was exported by a human at some point and is a fixed picture from then on.
The alternative is to apply the rules above to every match TBA has scored and
work the table out, which is what **Use TBA standings** does. That is not a
second table read off an endpoint: `web/lib/standings.js` accumulates RP, match
points, auto fuel, tower points and W-L-T per team from the score breakdowns,
drops surrogate appearances from the surrogate's own record, and re-sorts by the
official tiebreaker chain. `./sim.py report --standings tba` is the same thing on
the CLI.

The button says what it changed — how many teams' records differ and how many
rank positions moved — because most of the time the answer is "nothing", and the
one time it is not, that is exactly what you want to see before reading the
projection.

**TBA computes the same table from the same matches, so the rebuild is checked
against it.** The published rankings ride along in the bundle as `tbaRankings`
purely as a reference: the page compares its own rebuild — order, plus each
team's played / RP / W-L-T — and reports `matches TBA` or, in red, what
disagrees. A disagreement means the ranking rules here have drifted from the
real ones, which would quietly poison every projection; `./sim.py report
--standings tba` prints the same verdict, `tests/test_ranking.py` asserts it
across the fixture event, and the deploy workflow warns on it. If TBA has
published no rankings yet, the check simply says so.

The reference is never an input. Standings have to be *rebuildable* to be
projectable — the simulator adds hypothetical matches to them a few thousand
times over — so reading a finished table would not do.

## The model

Fitted on every played qualification match at the event:

| | |
|---|---|
| Hub OPR | ridge-regularised least squares on alliance hub points. Ridge shrinks toward the event mean, which matters at 27 teams / 70 alliance appearances |
| Auto fuel OPR | the same fit against `hubScore.autoPoints` |
| Recency | every observation is discounted by how long ago it happened, so a robot is rated on what it is doing now rather than on its whole event |
| Residual σ | spread of a single alliance's score around its prediction |
| Rating SE | how well each team's rating is pinned down; resampled per trial so a team with two matches is not treated as a known quantity |
| Climbs | per-robot empirical distribution over climb outcomes, smoothed toward a prior (scouted capability when available, else the event-wide pool) |
| Fouls | empirical draw of opponent-foul points |

### Recency

A robot mid-event is not a fixed quantity: intakes get fixed, hangers get
bolted on, drivers get better, and something breaks. So every alliance
appearance carries a weight that halves every **20 qualification matches** into
the past, and the fit minimises `Σ wᵢ (predictedᵢ − actualᵢ)² + ridge‖b − prior‖²`
instead of the flat sum. The weights are rescaled to average 1, which is what
keeps everything downstream on its usual scale — ridge strength, residual σ and
the standard errors all read against the number of observations, and a decay
that shrank the total weight would quietly turn into extra shrinkage.

Age is counted in **schedule position**, not in a team's own matches, so both
alliances of a match weigh the same and a team that has played more does not
decay faster. Only the gap between matches matters, so the weights are the same
whether they are computed mid-event or at the end.

It applies to everything that estimates *a team* — both OPR fits, their σ and
standard errors, the climb distribution, and the calibration of scouted climb
capability against observed climbs. It deliberately does **not** touch the RP
thresholds (a fixed rule of the game being read off the data, not something
that drifts) or the foul draw (a property of the opponent).

The half-life is the whole design decision, and it was not chosen on this event
alone. 2026auwarp's own backtest prefers a much shorter memory — half-life 6–12
scores best over its 20 out-of-sample matches — but replaying the same
walk-forward test over the reference regionals shows that setting is actively
harmful on a 66–80 match schedule, where each team's matches are spread thin
enough that a short half-life throws most of their record away:

```
                    2026auwarp        2026cosp        2026tuis        2026mndu
half-life        Brier    MAE      Brier    MAE     Brier    MAE     Brier    MAE
off (0)          0.157   28.7      0.112   60.0     0.169   45.1     0.178   35.9
6                0.139   27.5      0.145   65.5        —      —         —      —
20               0.150   28.1      0.114   61.6     0.167   45.7     0.181   35.6
```

20 is the longest half-life that still improves this event while leaving the
regionals where they were. A longer schedule wants a longer half-life:
`--half-life N` sets it, `--half-life 0` turns the decay off, and **Recency
half-life** in the UI does the same live. `./sim.py backtest` reports what it is
worth on whatever event you point it at, the same way it does for scouting.

Each trial draws a rating vector, then for every remaining match draws both
alliances' hub score (normal, truncated at zero, rounded — real scores are
integers, which is what makes a genuine tie possible), auto share, climbs and
fouls; awards RP by the rules above; adds it to the current standings; and
re-sorts by the official tiebreaker chain.

### Does it work?

`./sim.py backtest` refits the model at each point in the event and predicts the
next match it has not seen. On 2026auwarp's 20 out-of-sample matches, with the
default model (scouting on):

```
winner called       80%
Brier score         0.150   (0.25 = coin flip, lower is better)
mean score error    28.1 points per alliance
```

Calibration runs slightly *under*-confident — matches called at 60–70% all
landed — but on 20 matches that is within noise. See
[Does it help?](#does-it-help) for the same test without scouting.

## Pit scouting

The scouting app at https://scoutinapp.vercel.app
([source](https://github.com/Hazzer890/scoutinapp)) exposes a public read-only
dump of one event, added in `convex/http.ts`:

```
GET https://third-cow-432.convex.site/api/scouting[?event=<tbaKey>]
```

No key required. The deployment URL is the one the built app hands to
`ConvexReactClient` — HTTP actions live on `.convex.site` where the data API
lives on `.convex.cloud`. Override it with `--scouting-url`.

It is **pit** scouting — capability estimates filed once per team per scout, not
per-match performance — so it cannot replace the on-field fit. What it can do is
tell the fit what to believe about a team it has barely seen. For 2026auwarp
that is 315 reports over all 27 teams, plus 28 picklists.

### How it enters the model

Nothing is trusted at face value. Each scouted quantity is regressed onto the
on-field ratings, and **the r² of that regression becomes the weight it
carries** — so scouting that tracks results moves the fit, and scouting that
does not fades back to the plain event mean. The scale always comes from the
field even though the shape comes from the pit.

| Scouted | Feeds | Fit at this event |
|---|---|---|
| median balls per match | shrinkage target for hub OPR | r² 0.79 → trust 0.79 |
| median auto balls | shrinkage target for auto-fuel OPR | r² 0.50 → trust 0.50 |
| `canClimb` report rate | per-robot climb probability | calibrated against observed climbs |

Ridge regression previously pulled every team toward the event average, which
punishes a genuinely strong team with few matches. It now pulls each team toward
*what scouts expect of that team*. Climbs matter more than they look: six climbs
in seventy alliance appearances is not enough to tell a robot with a hanger from
one without, and Traversal is worth an RP. Scouts flagged exactly one team as a
climber (6509) — the only team that has climbed.

Two guards worth knowing about: a scouted non-climber keeps a 2% floor (pit
reports are a snapshot; hangers get bolted on mid-event), and per-team figures
are **medians** across scouts, so one report claiming 200 balls/match barely
moves anything.

### Does it help?

`./sim.py backtest` runs the walk-forward test both ways, same matches, same
trials, same seeds:

```
model                   called    Brier  log loss  score MAE
on-field only              80%    0.180     0.544       30.5
+ pit scouting             80%    0.150     0.476       28.1
```

Better probabilities and better score estimates out of sample. Turn it off with
`--no-scouting`, or the **Use pit scouting** checkbox in the UI; `Δ scouting` in
the model table shows what it moved for each team.

The same command runs the same test for the recency decay, everything else held
(`Δ recency` in the model table is its per-team counterpart):

```
model                   called    Brier  log loss  score MAE
every match equal          80%    0.157     0.491       28.7
half-life 20               80%    0.150     0.476       28.1
```

### Team ids

The scouting app keys teams by plain number, so it has the same quirk the CSV
does: `9982` is the team TBA calls `4788B` (nickname "Can't Control Pro Max" to
4788's "Can't Control"). The alias derived while reconciling the CSV is reused.
Two scouted teams — 9275 and 9975, two reports each — are not in the TBA event
and are ignored with a note in the UI.

## Data

- `../event-rankings.csv` — current standings, used as the starting point unless
  the [standings are rebuilt](#standings-the-csv-or-the-matches) from the matches.
- `cache/<event>/` — the TBA pull. Everything works offline from this cache;
  `--offline` never touches the network.
- `cache/<event>/scouting.json` — the scoutinapp dump, cached the same way.
- `cache/reference/` — six 2026 regionals, used once to pin down the RP rules.

The CSV lists one team as `9982` where TBA has `4788B`. Ids that appear on only
one side are matched by their ranking row (played / RP / W-L-T / match points)
rather than dropped, and the match is reported in the UI and on the CLI. Every
other field agrees exactly.

## No API

The UI is a thin client over three endpoints:

```
There is no API any more — the page does the work itself, and `sim.py serve` is
a plain static file server so local behaviour matches a static host exactly.

The old `POST /api/simulate` accepted up to 200,000 trials, and one request at
that size pinned a core for ~14 seconds: an unauthenticated CPU amplifier that
would have needed rate limiting and a queue before facing the internet. Moving
the sampler into the browser deleted the endpoint and the problem together.

For scripting, `./sim.py report --json` emits the same structure the endpoint
used to.
```

## Other events

```
./sim.py serve --event 2026xyz --csv path/to/standings.csv
```

The CSV is optional — without it the standings are rebuilt from TBA. Scouting is
optional too: if the export 404s for the event, or the app is down, the run falls
back to the on-field fit and says so. The RP rules are 2026-specific; the
thresholds adapt on their own.

`TBA_API_KEY` is read from the environment, falling back to `.env` here (which is
gitignored).

## Tests

```
python3 -m unittest discover -s tests
```
