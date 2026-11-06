---
name: xtrace-read
description: Read an Xtrace trace report of a CUDA kernel in the terminal with `xtrace show` — per-region statistics, per-thread timelines, pipeline bubbles, concurrency over time, outlier threads, the longest instances, and the region a user selected in the HTML panel. Use whenever there is a trace.json from a MarsCompute Xtrace job to analyse, or when the user pastes a selection token starting with gwtrace/1.
---

# Reading an Xtrace report

A trace job returns `trace.json` (the report), `report.html` (the same report as an
interactive page) and `summary.json` (the headline figures `xtrace_get_job` shows). Read
the report with `xtrace show`; it needs no GPU and no network.

## 1. Use `xtrace show`, not a JSON parser

The report carries a pre-computed analysis: paired intervals, per-region percentiles,
per-row active and bubble time, concurrency. `xtrace show` reads that analysis, and pairs
again from the raw records when a filter narrows the rows. A hand-written script
re-derives it all, and usually disagrees with the report in ways that look plausible:

- **Pairing.** `xtrace show` pairs chronologically: a leave closes the latest open enter
  of its row. One corrupted record then costs one interval. Pairing the i-th enter with
  the i-th leave cascades a single loss into inverted spans across the run.
- **Units.** An `sm` clock counts SM cycles and prints `cyc`. A script that prints `ns`
  is wrong by an unknown factor.
- **Rows.** A warp-level run's rows are warps. Counting them as threads is off by 32.
- **Filters.** A filter applies before aggregation, so a per-block number is that
  block's, not the run's.

```bash
xtrace show trace.json                    # header, region stats, timeline
xtrace show trace.json --stats            # per-site records, paired regions, per-region stats
xtrace show trace.json --bubbles          # active, bubble and lifetime per row, worst rows
xtrace show trace.json --timeline --sample 32 --sort-by bubble --human
xtrace show trace.json --concurrency      # rows active per time bin
xtrace show trace.json --outliers         # rows whose active time is an outlier
xtrace show trace.json --longest <REGION> # the single longest instance of a region
xtrace show trace.json --longest-bubble   # the largest gaps between intervals
xtrace show trace.json --json             # the views above, machine-readable
```

Filters compose with every view:

```bash
--tid '[0,31],[64,95]'    # rows by launch-wide thread id
--block 0-3,5             # by block
--warp 0,1                # by warp within its block
--stime <t> --etime <t>   # by time, in the report's ticks
--select '<token>'        # the region a user selected in the HTML panel
```

Read `--json` when you compute with the numbers; read the plain output when you read
them. Never parse the plain output. `--json` does not carry the per-site and
paired-region tables; read those from `--stats`.

## 2. Find the bubbles

A bubble is time a row spends between its regions: waiting on a barrier, on memory, on
another warp role. Work from the whole to the part:

1. **Is the trace trustworthy?** `--stats`: every region's pair rate, and records per
   row against the count the kernel's structure predicts (skill `xtrace-remote`, section
   5). Stop and fix the sites if either is off.
2. **How much is idle?** `--bubbles` gives the bubble ratio and the rows with the most
   bubble time. A ratio of a few percent is a busy kernel; tens of percent is a pipeline
   that stalls.
3. **When?** `--concurrency` shows whether idle time sits at the start (ramp-up, a
   prologue that loads before the loop), the end (tail: some blocks finish late) or
   throughout (a steady-state stall).
4. **Where?** `--longest-bubble` names the largest gaps, with the row and the regions on
   either side. A gap after a copy region and before a compute region is the compute
   warp waiting for data; the reverse is the producer waiting for a free buffer.
5. **Who?** Narrow with `--block` and `--warp`, then compare `--stats` across them. One
   slow block points at a placement or a data-dependent path; one slow warp role
   points at the pipeline's balance between roles.
6. **Which instance?** `--longest <REGION>` and `--outliers` find the row and the
   iteration to look at in `--timeline`.

Report bubbles as measurements: name the region pair, the rows, the duration and its
share of the row's lifetime, the view that shows it, and what the time unit is. Then
say what would explain it, and what trace would tell the explanations apart.

## 3. A selection token is the user pointing at the panel

`report.html` draws the trace as lanes over time. A user drags out a stretch of time and
a set of lanes, right-clicks, and copies one line:

```
gwtrace/1 kernel=square_matrix_mul#ef46 t=4365:11462ns lane=block:15-33 order=role scopes=mainloop
```

Hand it to the viewer verbatim:

```bash
xtrace show trace.json --select 'gwtrace/1 ...' --stats
```

- **Do not rewrite it into `--tid` or `--stime`.** Its lane ids count across the launch
  (a warp counts from the grid), while `--warp` counts within a block; rewriting it by
  hand answers about different rows.
- **`t` counts from the run's first interval**, not from the raw clock.
- **A refusal is information.** "copied from a panel over a different kernel" means the
  user is looking at another report. Ask which one; do not hunt for a matching region.

## 4. When unsure, ask the user to select

Ask the user to look at `report.html` and select the region when:

- two readings of the numbers both fit and imply different fixes;
- a count or duration contradicts the kernel's structure and you cannot tell whether the
  model or the trace is wrong;
- a pair rate is below 100% and you cannot say where the boundaries went;
- the user describes something they saw ("the middle is idle", "one warp lags") that the
  aggregates do not locate.

Ask concretely: name the file, what to look for, the gesture (drag the region,
right-click, copy the selection) and what you will do with it. Do not guess and proceed:
an analysis of the wrong region reports a real number about the wrong thing.

## 5. Related skills

- `xtrace-remote`: prepare, submit and gate the trace this report came from.
