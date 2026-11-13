---
name: marstrace-read
description: Read a MarsTrace trace report of a CUDA kernel in the terminal with `marstrace show` — per-region statistics, per-thread timelines, pipeline bubbles, concurrency over time, outlier threads, the longest instances, and the region a selection token names. Use whenever there is a marstrace.json from a MarsTrace job to analyse, or its result-manifest.json to judge, or when the user pastes a selection token starting with mtrace/1.
---

# Reading a MarsTrace report

A job's results land in `marstrace-results/<job>/` (`marstrace fetch`): `marstrace.json`
(the trace report) and `result-manifest.json` (its quality and reason codes). Read the
report with `marstrace show`; it needs no GPU and no network.

## 1. Use `marstrace show`, not a JSON parser

The report carries a pre-computed analysis: paired intervals, per-region percentiles,
per-row active and bubble time, concurrency. `marstrace show` reads that analysis, and pairs
again from the raw records when a filter narrows the rows. A hand-written script
re-derives it all, and usually disagrees with the report in ways that look plausible:

- **Pairing.** `marstrace show` pairs chronologically: a leave closes the latest open enter
  of its row. One corrupted record then costs one interval. Pairing the i-th enter with
  the i-th leave cascades a single loss into inverted spans across the run.
- **Units.** An `sm` clock counts SM cycles and prints `cyc`. A script that prints `ns`
  is wrong by an unknown factor.
- **Rows.** A warp-level run's rows are warps. Counting them as threads is off by 32.
- **Filters.** A filter applies before aggregation, so a per-block number is that
  block's, not the run's.

```bash
marstrace show marstrace.json                    # header, region stats, timeline
marstrace show marstrace.json --stats            # per-site records, paired regions, per-region stats
marstrace show marstrace.json --bubbles          # active, bubble and lifetime per row, worst rows
marstrace show marstrace.json --timeline --sample 32 --sort-by bubble --human
marstrace show marstrace.json --concurrency      # rows active per time bin
marstrace show marstrace.json --outliers         # rows whose active time is an outlier
marstrace show marstrace.json --longest <REGION> # the single longest instance of a region
marstrace show marstrace.json --longest-bubble   # the largest gaps between intervals
marstrace show marstrace.json --json             # the views above, machine-readable
```

Filters compose with every view:

```bash
--tid '[0,31],[64,95]'    # rows by launch-wide thread id
--block 0-3,5             # by block
--warp 0,1                # by warp within its block
--stime <t> --etime <t>   # by time, in the report's ticks
--select '<token>'        # the region a selection token names
```

Read `--json` when you compute with the numbers; read the plain output when you read
them. Never parse the plain output. `--json` does not carry the per-site and
paired-region tables; read those from `--stats`.

## 2. Gate the trace first

Read `quality` and `reason_codes` in `result-manifest.json` before any number of the
report. A job that ended without a trace is `unusable`: tell the user, and look at the
intent, the workspace and the caps of `[agent]` before submitting again; the job was
charged its usage all the same. `clock_type` and `granularity` say what a tick is (`gpu`, `sm` or `anchor`) and whether a row is a thread
or a warp.

`quality` is `usable`, `partial` or `unusable`. `reason_codes` says why it is not
`usable`; each code limits what the trace can support:

| Reason code | What it means for the reading |
| --- | --- |
| `no_trace_produced` | the session ended without a trace: no timeline exists. The other codes say why |
| `session_turn_limit_reached`, `session_time_limit_reached` | the agent stopped at `[agent] max_turns` or `max_wall_seconds`. What it published is complete, but it may not have placed every probe it meant to: check the regions against `intent`. Raise the cap before submitting again |
| `agent_session_incomplete` | the agent's session ended early for another reason; read what was published with the same care |
| `empty_trace` | nothing recorded: the sites never executed under this launch. No timeline exists |
| `sites_without_records` | some sites never executed on the traced path; regions using them are empty |
| `unpaired_region_boundaries` | a region's pair rate is below 100%: unequal enter and leave counts, or a record buffer that wrapped. Say which before using its numbers |
| `regions_without_intervals` | a declared region produced no interval: its sites are backwards or on disjoint paths |
| `record_limit_reached`, `trace_size_limit_reached` | the report was cut to fit: later records or runs are missing, so counts are lower bounds. Ask in `intent` for fewer sites per iteration, warp-level records or a smaller launch |
| `run_limit_reached`, `run_file_too_large` | some runs the agent saved were left out of the report |
| `non_sass_runs_skipped`, `mixed_granularity_runs_skipped`, `mixed_clock_runs_skipped`, `run_not_publishable`, `run_redaction_failed` | some runs were left out because they could not be published alongside the others; the report holds the rest |
| `out_of_range_records_dropped` | records outside the run's time range were dropped; the intervals that remain are sound |

A code not in this table is still a limit: quote it to the user rather than ignore it.

Then check the trace against the kernel's structure with `--stats`:

- **Records per row match your model.** A loop of N iterations gives N records per row at
  a loop-body site. A different count means the site sits elsewhere than its region
  assumes.
- **Every declared site appears**, and **spans make sense**: an interval close to the
  whole kernel is a wait the regions do not split out; a leave before its enter means the region
  is backwards.

## 3. Find the bubbles

A bubble is time a row spends between its regions: waiting on a barrier, on memory, on
another warp role. Work from the whole to the part:

1. **Is the trace trustworthy?** Section 2: quality, reason codes, and `--stats` against
   the kernel's structure. When any is off, tell the user before going on.
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

## 4. A selection token names a region

A selection token is one line that names a stretch of time and a set of lanes of one
kernel's trace. The user copies it with a right click on the trace in the console's job page,
as the token alone or as the whole `marstrace show` command:

```
mtrace/1 kernel=square_matrix_mul#ef46 t=4365:11462ns lane=block:15-33 order=role scopes=mainloop
```

When the user pastes one, hand it to the viewer verbatim, on the run it was copied from
(`--run <n>` for any run but the first, as the copied command already says):

```bash
marstrace show marstrace-results/<job>/marstrace.json --select 'mtrace/1 ...' --stats
```

The command reads the fetched result, so run `marstrace fetch <job>` first when it is not
there.

- **Do not rewrite it into `--tid` or `--stime`.** Its lane ids count across the launch
  (a warp counts from the grid), while `--warp` counts within a block; rewriting it by
  hand answers about different rows.
- **`t` counts from the run's first interval**, not from the raw clock.
- **A refusal is information.** "copied from a panel over a different kernel" means the
  token belongs to another report. Ask which one; do not hunt for a matching region.

When two readings of the numbers both fit and imply different fixes, or a count
contradicts the kernel's structure, say so and ask the user which rows and times they
mean. Do not guess and proceed: an analysis of the wrong region reports a real number
about the wrong thing.

## 5. Related skills

- `marstrace-remote`: prepare and submit the trace this report came from.
