---
name: marstrace-remote
description: Trace the timeline inside one CUDA kernel on an H200, B200 or B300 with MarsTrace, without a local GPU. Gather the kernel's cubins (from nvcc, Triton, torch.compile, TileLang or CuTeDSL output), and optionally its source, a launch description, input data and a driver script, into a workspace that an agent on the GPU host traces within the caps you set. Then estimate the cost, submit, fetch marstrace.json and read it. A job starts once the workspace's balance reaches a minimum and is charged the agent tokens it actually used, failed attempts included. Works through the `marstrace` CLI or its stdio MCP server (`marstrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel.
---

# Tracing a kernel with MarsTrace

MarsTrace records when chosen parts of a compiled kernel execute. A job hands a workspace
to an agent on the GPU host: the cubins, and optionally the source tree, a launch manifest,
tensors and a driver script. The agent reads them, reproduces the launch, traces the
kernel within the caps you set, and returns one `marstrace.json` that `marstrace show`
reads in the terminal (skill `marstrace-read`).

The flow: **settle the request, find the cubin, gather the workspace, set the caps,
estimate, submit, fetch, show.**

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/MarsTrace
echo "${MARSTRACE_API_KEY:?create one at console.marscompute.ai, MarsTrace keys: mt_live_...}"
marstrace hardware      # hardware ids, GPU, architecture, CUDA versions, token rates, agent model and limits
marstrace credit        # available balance, the minimum a job needs to start, the page to add credit
```

Every command has an MCP tool of the server `marstrace` (`marstrace mcp`, registered by
`.mcp.json`): `marstrace_hardware`, `marstrace_credit`, `marstrace_cubin_find`,
`marstrace_cubin_show`, `marstrace_estimate`, `marstrace_submit`, `marstrace_jobs`,
`marstrace_job`, `marstrace_wait`, `marstrace_cancel`, `marstrace_result`,
`marstrace_fetch`, `marstrace_show`. They take local paths and move the files themselves.
Use whichever is at hand; the steps below name both.

## 2. Settle the request

Settle these with the user before writing anything:

- **The hardware.** Pick a hardware id from `marstrace hardware`: `h200`, `b200` or
  `b300`, among those it shows as available. A cubin runs only on the SM version it was
  built for: H200 is `sm_90a`, B200 `sm_100a`, B300 `sm_103a`.
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
| CuTeDSL | have CuTeDSL keep its cubins in a directory (see its documentation on dumping compiled kernels), then `marstrace cubin find <name> --dir <that directory> --copy-to my-job` |
| only PTX at hand | `ptxas -arch=sm_90a -lineinfo -o k.cubin k.ptx`; MarsTrace takes cubins only |

A cached cubin is built for the GPU the user ran it on. Check with
`marstrace cubin show my-job/<file>.cubin`; when its target differs from the hardware,
rebuild for that target. `sm_103a` needs CUDA 12.9 or later.

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
  the launch from source and spends turns on it. State the problem size in `intent`.
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

## 5. Estimate, then submit

How MarsTrace charges, which the user must hear before the first submission:

- **A job starts only while the workspace's available balance is at least the minimum**
  that `marstrace credit` and the estimate show (set by MarsCompute, e.g. $100). While it
  runs, the job holds that minimum, so a balance of $150 with a $100 minimum runs one job
  at a time. The minimum is not the price.
- **The whole job is charged the agent tokens it actually used**: the input, output,
  cache-read and cache-write tokens of every attempt, retried and failed attempts
  included, at the token rates `marstrace hardware` shows. The GPU is not charged
  separately. A failure on MarsCompute's side is charged too, and MarsCompute corrects a
  charge for its own fault with a credit adjustment; a job whose agent never ran costs
  nothing. A cancelled job pays for the tokens it used up to the stop.
- **Usage is charged in full even past the available balance**; the balance can go
  negative. The cost at the bounds is the most the job can cost.

```bash
marstrace estimate my-job         # uploads, checks, prices: CUDA version, token rates, token bounds per attempt, cost at the bounds, balance and minimum
```

**Tell the user the token rates, the cost at the bounds (`at most`), the minimum and the available
balance, say that the job is charged the tokens it actually used over every attempt, failed ones
included, and get a yes before the first submission of a session.** After that, keep them
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
far, the amount held, and once the job ends the charge with a line per dimension. When it
fails, read its `error` (`code` and `message`) and each attempt's; tell the user what the
failed job was charged before submitting again.

## 6. Fetch and show

```bash
marstrace fetch <job>             # into marstrace-results/<job>/; --wait on submit does it
marstrace show marstrace-results/<job>/marstrace.json --stats
```

The result is `marstrace.json` and its `result-manifest.json`. Gate the trace before
reading conclusions from it (skill `marstrace-read`, section 2): the quality must be
`usable`, or each reason code explained. A session that ends without a trace has
`quality: "unusable"`; it was still charged its usage. Before submitting again, revisit the
intent, the workspace and the caps with the user. Results can be fetched for 30 days
after the job ends.

## 7. Related skills

- `marstrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
