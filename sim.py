#!/usr/bin/env python3
"""End-of-event FRC ranking simulator.

    ./sim.py report                  text projection for the default event
    ./sim.py report --n 20000        more trials
    ./sim.py report --win 9991       project with 9991 winning out
    ./sim.py export                  rebuild the static site's data bundle
    ./sim.py serve                   web UI on http://127.0.0.1:8765
    ./sim.py fetch                   force a fresh pull from TBA
    ./sim.py fit                     show the fitted team ratings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ranksim.event import rank_teams
from ranksim.loader import DEFAULT_CSV, DEFAULT_EVENT, load_event, load_fit, load_scouting
from ranksim.scouting import DEFAULT_URL as SCOUTING_URL
from ranksim.server import serve
from ranksim.simulate import SimOptions, simulate


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event", default=DEFAULT_EVENT, help="TBA event key")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="standings CSV")
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    parser.add_argument("--ridge", type=float, default=3.0, help="OPR ridge strength")
    parser.add_argument("--no-scouting", action="store_true", help="ignore the scoutinapp data")
    parser.add_argument("--scouting-url", default=SCOUTING_URL, help="scoutinapp export endpoint")
    parser.add_argument(
        "--scouting-weight", type=float, default=1.0,
        help="how far to trust scouting, 0-1 (scaled again by how well it predicts)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="refresh the local TBA cache")
    common_args(fetch)

    fitp = sub.add_parser("fit", help="print fitted team ratings")
    common_args(fitp)

    scout = sub.add_parser("scouting", help="print the pulled scouting data")
    common_args(scout)

    report = sub.add_parser("report", help="run a simulation and print a projection")
    common_args(report)
    report.add_argument("-n", "--n", type=int, default=5000, help="number of trials")
    report.add_argument("--seed", type=int, default=None)
    report.add_argument("--cutoff", type=int, default=8, help="alliance-captain cutoff")
    report.add_argument("--no-uncertainty", action="store_true", help="fix team ratings at their point estimates")
    report.add_argument("--win", action="append", default=[], metavar="TEAM", help="force TEAM to win every remaining match (repeatable)")
    report.add_argument("--lose", action="append", default=[], metavar="TEAM", help="force TEAM to lose every remaining match (repeatable)")
    report.add_argument("--force", action="append", default=[], metavar="MATCH=COLOR", help="force a match, e.g. 2026auwarp_qm40=red")
    report.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")

    back = sub.add_parser("backtest", help="walk-forward check of the match predictions")
    common_args(back)
    back.add_argument("--min-matches", type=int, default=15, help="matches to fit on before the first prediction")
    back.add_argument("--trials", type=int, default=800, help="trials per predicted match")

    exp = sub.add_parser("export", help="write the static site's event bundle")
    common_args(exp)
    exp.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "web" / "data" / "event.json"
    )

    serve_p = sub.add_parser("serve", help="run the web UI")
    common_args(serve_p)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8765)

    return parser


def resolve_forced(state, args) -> dict[str, str]:
    forced: dict[str, str] = {}
    for spec in args.force:
        if "=" not in spec:
            raise SystemExit(f"--force expects MATCH=COLOR, got {spec!r}")
        key, color = spec.split("=", 1)
        if color not in ("red", "blue", "tie"):
            raise SystemExit(f"--force colour must be red/blue/tie, got {color!r}")
        if not key.startswith(state.event_key):
            key = f"{state.event_key}_qm{key}" if key.isdigit() else key
        forced[key] = color
    for team in args.win:
        for match in state.remaining:
            if team in match.red:
                forced[match.key] = "red"
            elif team in match.blue:
                forced[match.key] = "blue"
    for team in args.lose:
        for match in state.remaining:
            if team in match.red:
                forced[match.key] = "blue"
            elif team in match.blue:
                forced[match.key] = "red"
    return forced


def print_fit(state, fit) -> None:
    print(f"{state.event_name} ({state.event_key})")
    print(
        f"  fitted on {fit.n_observations} alliance-appearances, ridge {fit.ridge}, "
        f"R^2 {fit.r_squared:.3f}, residual sigma {fit.sigma_hub:.1f} hub pts"
    )
    info = fit.scouting
    if info.get("used"):
        print(
            f"  scouting priors from {info['reports']} pit reports "
            f"(hub trust {info['hub']['trust']:.2f}, auto trust {info['auto']['trust']:.2f})"
        )
    else:
        print("  no scouting priors (on-field data only)")
    for name, value in fit.thresholds.items():
        print(f"  {name:<13} >= {value:g}  ({fit.threshold_sources[name]})")
    print()
    print(f"{'team':>7}  {'hub OPR':>8}  {'+/-':>5}  {'auto OPR':>8}  {'climb %':>7}  {'E[tower]':>8}")
    for team in sorted(state.teams, key=lambda t: -fit.hub[t]):
        print(
            f"{team:>7}  {fit.hub[team]:8.1f}  {fit.hub_se[team]:5.1f}  {fit.auto[team]:8.1f}"
            f"  {100 * fit.tower_rate(team):6.1f}%  {fit.expected_tower(team):8.1f}"
        )


def print_scouting(state, scouting, fit) -> None:
    if scouting is None:
        print("no scouting data available")
        return
    print(f"{scouting.event_name} ({scouting.event_key}) -- scoutinapp")
    print(
        f"  {scouting.total_reports} pit reports over "
        f"{sum(1 for s in scouting.teams.values() if s.reports)} teams, "
        f"{scouting.picklists} picklists"
    )
    for note in scouting.unmatched:
        print(f"  not at this event, ignored: {note}")
    info = fit.scouting
    if info.get("used"):
        for key in ("hub", "auto"):
            block = info[key]
            state_word = "applied" if block.get("applied") else "not applied"
            print(
                f"  {block['metric']:<22} -> r^2 {block['rSquared']:.2f}, "
                f"trust {block['trust']:.2f} ({state_word})"
            )
        climb = info["climb"]
        print(f"  {climb['metric']:<22} -> {climb['mode']}")
    print()
    header = (
        f"{'team':>7}  {'reps':>4}  {'balls':>6}  {'auto':>5}  {'climb':>6}  "
        f"{'driver':>6}  {'def':>4}  {'pick':>5}  {'hubOPR':>7}"
    )
    print(header)
    print("-" * len(header))
    for team in sorted(state.teams, key=lambda t: -fit.hub[t]):
        s = scouting.teams.get(team)
        if s is None:
            continue
        pick = f"{s.picklist_rank:.0f}" if s.picklist_rank is not None else "-"
        print(
            f"{team:>7}  {s.reports:>4}  {s.balls_per_match:>6.0f}  {s.auto_balls:>5.0f}"
            f"  {100 * s.can_climb_rate:>5.0f}%  {s.driver_rating:>6.1f}  {s.defense_rating:>4.1f}"
            f"  {pick:>5}  {fit.hub[team]:>7.1f}"
        )


def print_report(state, result, fit=None) -> None:
    meta = result["meta"]
    print(f"{state.event_name} ({state.event_key})")
    print(
        f"  {len(state.played)} played / {meta['remainingMatches']} remaining, "
        f"{meta['n']} trials, top-{meta['cutoff']} cutoff"
    )
    if meta["forced"]:
        print(f"  scenario: {len(meta['forced'])} match(es) forced")
    if state.csv_discrepancies:
        print(f"  note: {len(state.csv_discrepancies)} CSV/TBA discrepancies (see `fit`)")
    if fit is not None:
        info = fit.scouting
        print(
            f"  model: scouting priors from {info['reports']} pit reports"
            if info.get("used")
            else "  model: on-field data only"
        )
    print()
    header = (
        f"{'now':>3}  {'team':>7}  {'RP':>4}  {'RS':>5}  {'proj':>5}  {'range':>9}  "
        f"{'P(#1)':>6}  {'P(top%d)' % meta['cutoff']:>8}  {'E[RP]':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in result["teams"]:
        cur = row["current"]
        print(
            f"{row['currentRank']:>3}  {row['team']:>7}  {cur['rp']:>4}  {cur['rankScore']:>5.2f}"
            f"  {row['meanRank']:>5.1f}  {row['p05Rank']:>4}-{row['p95Rank']:<4}"
            f"  {100 * row['pRank1']:>5.1f}%  {100 * row['pCutoff']:>7.1f}%  {row['expectedRp']:>6.1f}"
        )
    print()
    print("remaining matches")
    for m in result["matches"]:
        flag = f"  [forced {m['forced']}]" if m["forced"] else ""
        print(
            f"  qm{m['number']:<3} red {'-'.join(m['red']):<20} vs blue {'-'.join(m['blue']):<20}"
            f"  red {100 * m['pRed']:5.1f}% / blue {100 * m['pBlue']:5.1f}%{flag}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        # Export first, so the served bundle always matches the local cache --
        # the page reads data/event.json, not the Python objects.
        import time

        from ranksim.export import write

        state = load_event(args.event, csv_path=args.csv, offline=args.offline)
        out = Path(__file__).parent / "web" / "data" / "event.json"
        write(
            state,
            out,
            ridge=args.ridge,
            scouting_url=args.scouting_url,
            generated_at=time.time(),
        )
        print(f"{state.event_name} ({state.event_key})")
        print(
            f"  {len(state.played)} played, {len(state.remaining)} remaining, "
            f"{len(state.teams)} teams -> {out.relative_to(Path(__file__).parent)}"
        )
        print("  scouting is fetched live by the page; use the Refresh scouting button")

        from ranksim.server import RefreshService

        service = RefreshService(
            args.event, args.csv, args.ridge, args.scouting_url, out
        )
        serve(host=args.host, port=args.port, service=service)
        return 0

    state = load_event(
        args.event, csv_path=args.csv, offline=args.offline, refresh=args.command == "fetch"
    )
    scouting, scouting_error = (None, None)
    if not args.no_scouting:
        scouting, scouting_error = load_scouting(
            state,
            url=args.scouting_url,
            offline=args.offline,
            refresh=args.command == "fetch",
        )
    fit = load_fit(
        state, ridge=args.ridge, scouting=scouting, scouting_weight=args.scouting_weight
    )
    if scouting_error:
        print(f"  scouting unavailable, using on-field data only: {scouting_error}")

    if args.command == "fetch":
        print(
            f"cached {state.event_name}: {len(state.played)} played, "
            f"{len(state.remaining)} remaining, {len(state.teams)} teams"
        )
        for note in state.csv_discrepancies:
            print(f"  CSV/TBA: {note}")
        for csv_team, tba_team in state.csv_aliases.items():
            print(f"  CSV team {csv_team} matched to TBA team {tba_team}")
        return 0

    if args.command == "backtest":
        from ranksim import backtest

        out = backtest.run(
            state,
            ridge=args.ridge,
            min_matches=args.min_matches,
            trials=args.trials,
            scouting=scouting,
            scouting_weight=args.scouting_weight,
        )
        print(f"{state.event_name} ({state.event_key})")
        print(
            f"  {out['n']} out-of-sample matches, each predicted from only the matches before it"
        )
        print(f"  winner called       {100 * out['accuracy']:.0f}%")
        print(f"  Brier score         {out['brier']:.3f}  (0.25 = coin flip, lower is better)")
        print(f"  log loss            {out['logLoss']:.3f}")
        print(f"  mean score error    {out['scoreMae']:.1f} points per alliance")
        if out["calibration"]:
            print()
            print(f"  {'confidence':<12}{'n':>4}{'predicted':>11}{'actual':>9}")
            for b in out["calibration"]:
                print(
                    f"  {b['range']:<12}{b['n']:>4}{100 * b['predicted']:>10.0f}%{100 * b['actual']:>8.0f}%"
                )
        if scouting is not None:
            # Same matches, same trials, same seeds -- the only difference is
            # whether the scouting priors were in the fit.
            plain = backtest.run(
                state,
                ridge=args.ridge,
                min_matches=args.min_matches,
                trials=args.trials,
                scouting=None,
            )
            print()
            print("  does scouting help?")
            print(f"  {'model':<22}{'called':>8}{'Brier':>9}{'log loss':>10}{'score MAE':>11}")
            for label, res in (("on-field only", plain), ("+ pit scouting", out)):
                print(
                    f"  {label:<22}{100 * res['accuracy']:>7.0f}%{res['brier']:>9.3f}"
                    f"{res['logLoss']:>10.3f}{res['scoreMae']:>11.1f}"
                )
        print()
        # Scores are ranking match points, which exclude points awarded from the
        # opponent's fouls -- so an alliance can win a match it trails on here.
        print(f"  {'match':<8}{'p(red)':>8}{'result':>9}{'match pts':>12}")
        for p in out["predictions"]:
            mark = "" if not p.actual else ("  ok" if p.called else "  miss")
            print(
                f"  qm{p.match_number:<6}{100 * p.p_red:>7.0f}%{p.actual or 'tie':>9}"
                f"{f'{p.red_score}-{p.blue_score}':>12}{mark}"
            )
        return 0

    if args.command == "export":
        import time

        from ranksim.export import write

        path = write(
            state,
            args.out,
            ridge=args.ridge,
            scouting_url=args.scouting_url,
            generated_at=time.time(),
        )
        size = path.stat().st_size
        print(f"wrote {path} ({size / 1024:.0f} KB)")
        print(
            f"  {len(state.played)} played matches, {len(state.remaining)} remaining, "
            f"{len(state.teams)} teams, {len(state.results)} alliance appearances"
        )
        print(f"  scouting is fetched live by the page from {args.scouting_url}")
        return 0

    if args.command == "scouting":
        print_scouting(state, scouting, fit)
        return 0

    if args.command == "fit":
        print_fit(state, fit)
        for note in state.csv_discrepancies:
            print(f"  CSV/TBA: {note}")
        for csv_team, tba_team in state.csv_aliases.items():
            print(f"  CSV team {csv_team} matched to TBA team {tba_team}")
        return 0

    options = SimOptions(
        n=args.n,
        seed=args.seed,
        opr_uncertainty=not args.no_uncertainty,
        cutoff=args.cutoff,
        forced=resolve_forced(state, args),
    )
    result = simulate(state, fit, options)
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print_report(state, result, fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
