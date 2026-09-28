"""Wall-clock time of a run, minus the time the machine was suspended.

Start and end come from the first and last timestamp in the run's main.log.
Suspend periods come from the systemd journal ("PM: suspend entry" / "PM: suspend exit").
If the journal of that time is gone, save it once with
    journalctl -o short-iso | grep "PM: suspend" > suspends.txt
and pass --journal suspends.txt.

Usage: python utils/run_time.py logs/robocasa/2026-09-20/R_sir_seed43_train logs/robocasa/2026-09-20/R_sir_seed43_eval
"""
import argparse
import os
import re
import subprocess
from datetime import datetime


def log_start_end(run_dir):
    times = []
    for line in open(os.path.join(run_dir, "main.log")):
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if m:
            times.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    return times[0], times[-1]


def suspends(since, until, journal_file=None):
    if journal_file:
        lines = open(journal_file).read().splitlines()
    else:
        out = subprocess.run(["journalctl", "_TRANSPORT=kernel", "-o", "short-iso",
                              "--since", f"{since:%Y-%m-%d %H:%M:%S}", "--until", f"{until:%Y-%m-%d %H:%M:%S}",
                              "--grep", "PM: suspend"],
                             capture_output=True, text=True).stdout
        lines = out.splitlines()

    periods = []
    start = None
    for line in lines:
        if not line[:1].isdigit():  # "-- Boot ..." separators
            continue
        t = datetime.fromisoformat(line.split()[0]).replace(tzinfo=None)
        if "suspend entry" in line:
            start = t
        elif "suspend exit" in line and start is not None:
            periods.append((start, t))
            start = None
    return periods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories containing main.log")
    ap.add_argument("--journal", help="saved journalctl output instead of reading the live journal")
    args = ap.parse_args()

    runs = {run: log_start_end(run) for run in args.runs}
    since = min(start for start, _ in runs.values())
    until = max(end for _, end in runs.values())
    sleep = suspends(since, until, args.journal)

    for run, (start, end) in runs.items():
        wall = (end - start).total_seconds()

        asleep = []
        for s, e in sleep:
            overlap = (min(e, end) - max(s, start)).total_seconds()
            if overlap > 0:
                asleep.append((s, e, overlap))
        total = sum(o for _, _, o in asleep)

        print(f"{os.path.basename(os.path.normpath(run))}: "
              f"{wall / 3600:.2f} h wall-clock, {total / 3600:.2f} h suspended, "
              f"{(wall - total) / 3600:.2f} h running")
        for s, e, o in asleep:
            print(f"    suspended {s:%m-%d %H:%M} - {e:%H:%M} ({o / 60:.0f} min)")


if __name__ == "__main__":
    main()
