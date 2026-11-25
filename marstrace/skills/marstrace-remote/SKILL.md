---
name: marstrace-remote
description: Trace the timeline inside one CUDA kernel on an H200 or B200 with MarsTrace, without a local GPU. Gather the kernel's cubins (from nvcc, Triton, torch.compile, TileLang or CuTeDSL output), and optionally its source, a launch description, input data and a driver script, into a workspace that an agent on the GPU host traces within the caps you set. Then estimate the cost, submit, fetch marstrace.json and read it; or, in the thinking modes, have the agent optimize the kernel against your run and verify commands and fetch the patch and reports. A job starts once the workspace's balance reaches a minimum and is charged the agent tokens it actually used, failed attempts included, or nothing when its trace is unusable or it fails without a result. Works through the `marstrace` CLI or its stdio MCP server (`marstrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel.
---

# Tracing a kernel with MarsTrace

MarsTrace records when chosen parts of a compiled kernel execute. A job hands a workspace
to an agent on the GPU host: the cubins, and optionally the source tree, a launch manifest,
tensors and a driver script. The agent reads them, reproduces the launch, traces the
kernel within the caps you set, and returns one `marstrace.json` that `marstrace show`
reads in the terminal (skill `marstrace-read`).

The flow: **settle the request, find the cubin, gather the workspace, set the caps,
estimate, submit, fetch, show.**

## 1. Set up the account, the key and the client

MarsTrace runs on the user's MarsCompute account and is paid from its prepaid credit. When
`MARSTRACE_API_KEY` is unset, or a call answers `api_key_missing` or
`authentication_required`, walk the user through these steps. They happen in the user's
browser and shell; the agent cannot do them:

1. **Register** at https://console.marscompute.ai/register with an email code or Google.
   There is no approval; the new workspace is active at once.
2. **Save the billing profile** at https://console.marscompute.ai/tenant/billing/profile:
   legal name, country and address.
3. **Add credit** at https://console.marscompute.ai/tenant/billing/credits, by card. A job
   starts only while the available balance reaches the minimum that `marstrace credit`
   shows (section 5), so add at least that.
4. **Create a MarsTrace key** at https://console.marscompute.ai/tenant/trace/keys. It starts
   with `mt_live_` and is shown once. The console's other API keys, for its models, do not
   work with MarsTrace.
5. **Hand the key to the client** in the shell that starts the agent, e.g.
   `export MARSTRACE_API_KEY=mt_live_...` in the shell profile, then restart the agent: the
   MCP server reads the key from its environment. Never ask the user to paste the key into
   the conversation, and never write it into a file of the job directory.

Then check the client and the account:

```bash
pip install 'git+https://github.com/mars-compute-ai/MarsTrace@v0.3.0'   # the version the README installs
marstrace hardware      # hardware ids, GPU, architecture, CUDA versions, token rates, agent model and limits
marstrace credit        # available balance, the minimum a job needs to start, the page to add credit
```

`403 forbidden` ("not active, or MarsTrace is not enabled") means the workspace cannot run
MarsTrace yet: the user should check the console, and contact MarsCompute if it persists.

Every command has an MCP tool of the server `marstrace` (`marstrace mcp`, registered by
`.mcp.json`): `marstrace_guide`, `marstrace_hardware`, `marstrace_credit`,
`marstrace_cubin_find`, `marstrace_cubin_show`, `marstrace_estimate`, `marstrace_submit`,
`marstrace_jobs`, `marstrace_job`, `marstrace_wait`, `marstrace_cancel`,
`marstrace_result`, `marstrace_fetch`, `marstrace_show`. They take local paths and move the
files themselves. Use whichever is at hand; the steps below name both. `marstrace guide`
(MCP: `marstrace_guide`) prints this guide and the reading guide, for an agent without the
plugin.

## 2. Settle the request

Settle these with the user before writing anything:

- **The hardware.** Pick a hardware id from `marstrace hardware`: `h200` or `b200`, among
  those it shows as available; `marstrace hardware` lists the hardware the account can use.
  A cubin runs only on the SM version it was built for: H200 is `sm_90a`, B200 `sm_100a`.
- **The CUDA version.** The agent's CUDA toolkit is one of the versions `marstrace hardware`
  lists for the hardware, the one marked default unless `cuda_version` names another. Pick
  the version the user builds with, or one at least as new as the toolkit that built the
  cubin.
- **The kernel and the launch that matters.** A trace answers for the launch it observed:
  grid, block, problem size and data layout all change the timeline. Ask for the shape
  when the code leaves it open.
- **What to time.** Phases (load, compute, store, the wait on a barrier), warp roles, or
  per-iteration cost. Write it in `intent`, with the problem size; it is the agent's task.

## 3. Find the cubin

| Where the kernel comes from | How to get its cubin |
| --- | --- |
| CUDA C++ | `nvcc -cubin -arch=sm_90a -lineinfo -o k.cubin k.cu` (no GPU needed) |
| Triton, torch.compile | run it once, then `marstrace cubin find <name> --copy-to my-job` |
| TileLang | run it once, then `marstrace cubin find <name> --copy-to my-job` |
| CuTeDSL | compile it with `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=<dir> CUTE_DSL_LINEINFO=1` (older releases: `CUTE_DSL_KEEP_CUBIN=1`); `CUTE_DSL_ARCH=sm_100a` targets another GPU and compiles without one. Then `marstrace cubin find <name> --dir <dir> --copy-to my-job` |
| only PTX at hand | `ptxas -arch=sm_90a -lineinfo -o k.cubin k.ptx`; MarsTrace takes cubins only |

A cached cubin is built for the GPU the user ran it on. Check with
`marstrace cubin show my-job/<file>.cubin`; when its target differs from the hardware,
rebuild for that target.

For Triton, `marstrace cubin find` prints what the cache recorded: the block size
(`num_warps × 32`), dynamic shared memory, cluster shape and the kernel's signature. Put
those facts in `intent` or a driver script, so the agent does not spend turns finding them.

## 4. Gather the workspace

```
my-job/
  marstrace.toml
  k.cubin                 the kernel, built for the hardware's architecture
  src/                    optional: the source tree; archived without .git
  launch.json             optional: a marscompute.kernel-launch manifest of the launch
  inputs.safetensors      optional: real input data
  run.py                  optional: a host driver script, a build file
```

```toml
hardware = "b200"                 # a hardware id from marstrace hardware
cuda_version = "13.0"             # optional: a CUDA version the hardware offers; its default when absent
intent = "find the longest bubble in the mainloop of the producer warps at M=N=K=8192"

[workspace]
cubins = ["k.cubin"]              # 1 to 8, each at most 64 MiB
source = "src"                    # optional: a directory, uploaded as source.tar.gz
manifest = "launch.json"          # optional
tensors = ["inputs.safetensors"]  # 0 to 8, each at most 128 MiB
files = ["run.py"]                # 0 to 8, each at most 16 MiB: driver scripts, build files

[agent]                           # optional; an absent key takes the default of marstrace hardware
max_turns = 200                   # 1 to 500
max_input_tokens = 20000000       # 1 to 50,000,000, cache reads and writes included
max_output_tokens = 200000        # 1 to 1,000,000
max_wall_seconds = 2400           # 60 to 2580: how long the agent works

[limits]                          # optional
max_attempts = 2                  # 1 to 3, 2 when absent: how often a failed attempt is retried
```

- **Give the agent a way to run the launch.** A driver script with the shapes the user
  cares about, a launch manifest, or source the agent can build; without one it derives
  the launch from source and spends turns on it. State the problem size in `intent`. A
  driver script that launches the kernel as the user does is the surest of the three.
- **A CuTeDSL kernel takes its arguments as a struct its host code builds** (TMA
  descriptors, layouts), so a driver must run that host code. Export it with the
  compiled function's `export_to_c(dir, name)`, link the object and the DSL's
  `libcuda_dialect_runtime_static.a` into a small C driver with `-lcuda`, and send the
  driver, the object and the archive in `files`: the driver then runs with the CUDA
  driver alone. The object embeds the cubin, so send the same cubin in `cubins`.
- **A launch manifest** is JSON whose `schema_version` is `marscompute.kernel-launch`. The
  agent on the GPU host reads the rest as a description, so name the kernel, the launch
  shape, the buffers and the arguments plainly:

  ```json
  {
    "schema_version": "marscompute.kernel-launch",
    "kernel": "square_matrix_mul",
    "grid": [16, 16, 1],
    "block": [16, 16, 1],
    "dynamic_shared_memory_bytes": 0,
    "buffers": [
      {"name": "a", "dtype": "int32", "shape": [256, 256], "init": "random"},
      {"name": "b", "dtype": "int32", "shape": [256, 256], "init": "random"},
      {"name": "c", "dtype": "int32", "shape": [256, 256], "init": "zeros"}
    ],
    "arguments": ["a", "b", "c", 256]
  }
  ```
- **Real data only where it steers control flow.** Random inputs time a GEMM or an
  attention kernel as well as real ones. Send tensors for routing tables, block tables,
  sequence lengths and anything else the kernel branches on.
- **The workspace is shared on purpose.** The agent reads, builds and runs the source
  tree and the files. Leave out what the user does not want to share: secrets, data
  beyond the launch, unrelated code. `.git` is never uploaded. All inputs of a job hold
  at most 256 MiB together.
- **Size the caps.** The agent's turns and tokens cap its session, and `max_wall_seconds`
  bounds it in time. The token caps and `max_attempts` are the only bounds on what a job
  costs: there is no dollar cap. The job is charged the tokens it actually uses, usually far
  less than the caps.

A job's `mode` is `fast` (the default): it delivers the trace and the analyser's summary of
it, with where each finding sits. `thinking-light` and `thinking-heavy` trace the kernel the
same way, then optimize it (section 4a).

## 4a. Optimize a kernel

Use `thinking-light` or `thinking-heavy` when the user wants the kernel faster, not only
explained. The agent changes the source one measured iteration at a time and keeps a change
only when it is faster and `verify` still passes. `thinking-heavy` has a larger budget and
needs a larger minimum balance (section 5).

```toml
mode = "thinking-light"           # or thinking-heavy

[workspace]
cubins = ["k.cubin"]              # traced first, as in a fast job
source = "src"                    # required: the tree the agent builds, runs and changes

[optimize]                        # required in these modes, refused in fast
build = "cmake -S . -B build && cmake --build build"   # optional
run = "python3 bench.py"          # required: launches the target kernel at its workload
verify = "python3 check.py"       # required: exits 0 when the results are correct
kernel = "flash_fwd.*"            # optional: a regular expression of the target kernel's name
max_turns = 400                   # optional caps of the optimization stage; light: 1 to 1,000
max_input_tokens = 40000000       # light: 1 to 60,000,000; heavy: 1 to 250,000,000
max_output_tokens = 400000        # light: 1 to 1,000,000; heavy: 1 to 3,000,000
max_wall_seconds = 3600           # light: 600 to 3,600; heavy: 600 to 18,000
```

- **The commands run with `bash -lc` in the root of `source`**, each 1 to 4,096 characters.
  Check them with the user: `run` must launch the shapes they care about, and `verify` must
  fail on a wrong result, at edge sizes too. The agent trusts `verify` alone for correctness.
- **There is no network at build or run time.** Every dependency the image lacks ships in
  `source`: vendored headers, wheels, data. Say so to the user before the estimate; the
  estimate repeats it as a note.
- **The caps add up.** An attempt's bounds are `[agent]` (the trace) plus `[optimize]` (the
  optimization); `[limits] max_attempts` defaults to 2 for light and 3 for heavy. Absent caps
  take the tier's defaults: light 400 turns, 40M input, 400k output tokens, 3,600 s; heavy
  2,000 turns, 150M, 1.5M, 18,000 s. Lower them when the task is small.
- **`examples/matmul-optimize`** in the client's repository is a small working layout.

With MCP, `marstrace_estimate` and `marstrace_submit` take `mode` and `optimize` (an object
of the same keys) to override the manifest.

## 5. Estimate, then submit

How MarsTrace charges, which the user must hear before the first submission:

- **A job starts only while the workspace's available balance is at least the minimum**
  of its mode, which the estimate shows: $100 for `fast` and `thinking-light`, $300 for
  `thinking-heavy` (set by MarsCompute; `marstrace credit` shows the base minimum). While it
  runs, the job holds that minimum, so a balance of $150 with a $100 minimum runs one job
  at a time. The minimum is not the price.
- **The whole job is charged the agent tokens it actually used**: the input, output,
  cache-read and cache-write tokens of every attempt, retried and failed attempts
  included, at the token rates `marstrace hardware` shows. Most of an agent's tokens are
  cache reads, so quote all five rates, not the input and output rates alone. The GPU is not charged
  separately. A job whose trace is delivered `unusable` costs nothing, whatever the cause,
  and so do a job that fails without delivering a result and a job whose agent never ran.
  A job a retry finished is charged its failed attempts too, and MarsCompute corrects a
  charge for its own fault with a credit adjustment. A cancelled
  job pays for the tokens it used up to the stop.
- **Usage is charged in full even past the available balance**; the balance can go
  negative. The cost at the bounds is the most the job can cost.

```bash
marstrace estimate my-job         # uploads, checks, prices: CUDA version, token rates, token bounds per attempt, cost at the bounds, balance and minimum
```

**Tell the user the token rates, the cost at the bounds (`at most`), the minimum and the available
balance, say that the job is charged the tokens it actually used over every attempt, failed ones
included, and nothing if its trace is unusable or it fails without a result, and get a yes before the first submission of a session.** After that, keep them
informed of what each job was charged. An estimate marked refused means the available
balance is below the minimum or billing is not active: give the user the page where credit
is added.

```bash
marstrace submit my-job --wait    # prices again, asks, submits, waits, fetches
marstrace submit my-job --yes     # when the user already agreed to the cost
```

With MCP: `marstrace_estimate`, then `marstrace_submit` with `accept_cost_at_bounds_usd`
set to the cost at the bounds the user agreed to (it refuses a higher one, since prices are
read again at submission), then `marstrace_wait` until `status` is `succeeded`, `failed` or
`cancelled`. Pass an `idempotency_key` (CLI: `--idempotency-key`); when a submission's
answer is lost, submit the same directory again with the same key: the client resends the
recorded request, and the service answers with the job it already started.

- **Keep the caps near what the task needs** with `[agent]` and `[limits] max_attempts`,
  so the cost at the bounds the user agrees to stays small. One attempt costs less at its
  bounds; a second one retries a failure at the price of what both use.
- **`insufficient_credit`, `spending_limit_exceeded` and `billing_suspended`** mean the
  balance is below the minimum or billing is not active; they name the console page where
  credit is added. Give it to the user. Do not retry.
- **`too_many_jobs`** means the workspace already has its maximum of queued jobs; wait the
  `retry after` time, or for one to start.
- **`marstrace cancel <job>`** stops a job the user no longer wants; it is charged what its
  attempts used up to the stop.

Read the job with `marstrace job <id>`: each attempt's status and failure, the usage so
far, the amount held, and once the job ends the charge with a line per dimension. A thinking
job adds an `optimize` line once its first iteration is done (stage, iteration, changes kept
and reverted, best speedup, seconds against the budget), and an `optimized` line when it
succeeds; `marstrace_job` answers them as `progress.optimize` and `result.optimization`. When it
fails, read its `error` (`code` and `message`) and each attempt's; a failed job without a
result is not charged (`cost.not_charged_reason` is `failed_no_result`), so tell the user it
cost nothing before submitting again.

## 6. Fetch and show

```bash
marstrace fetch <job>             # into marstrace-results/<job>/; --wait on submit does it; prints the analysis
marstrace show marstrace-results/<job>/marstrace.json --analysis   # the delivered analysis again
marstrace show marstrace-results/<job>/marstrace.json --stats
marstrace show marstrace-results/<job>/marstrace.json --optimization   # a thinking job's result again
```

A thinking job's fetch also writes `optimized.patch`, the reports and each iteration's files
to `marstrace-results/<job>/optimization/`, and prints the speedup per configuration.
`marstrace_fetch` answers them as `optimization`, `optimization_text` and
`optimization_files`. Skill `marstrace-read`, section 2b, says how to read them.

The user can also watch the job and read its trace in the console, at
https://console.marscompute.ai/tenant/trace/jobs; a region marked there and copied with a
right click is a `marstrace show --select` command (skill `marstrace-read`, section 4).

The result is `marstrace.json` and its `result-manifest.json`. Gate the trace before
reading conclusions from it (skill `marstrace-read`, section 2): the quality must be
`usable`, or each reason code explained. A session that ends without a trace has
`quality: "unusable"` and is not charged. Before submitting again, revisit the
intent, the workspace and the caps with the user. Results can be fetched for 30 days
after the job ends.

## 7. Related skills

- `marstrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
