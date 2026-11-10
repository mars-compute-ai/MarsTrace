---
name: marstrace-remote
description: Trace the timeline inside one CUDA kernel on an H200, B200 or B300 with MarsTrace, without a local GPU. Gather the kernel's cubins (from nvcc, Triton, torch.compile, TileLang or CuTeDSL output), and optionally its source, a launch description, input data and a driver script, into a workspace that an agent on the GPU host traces within a budget you set. Then estimate the cost, submit, fetch marstrace.json and read it. Works through the `marstrace` CLI or its stdio MCP server (`marstrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel.
---

# Tracing a kernel with MarsTrace

MarsTrace records when chosen parts of a compiled kernel execute. A job hands a workspace
to an agent on the GPU host: the cubins, and optionally the source tree, a launch manifest,
tensors and a driver script. The agent reads them, reproduces the launch, traces the
kernel within the budget you set, and returns one `marstrace.json` that `marstrace show`
reads in the terminal (skill `marstrace-read`).

The flow: **settle the request, find the cubin, gather the workspace, set the budget,
estimate, submit, fetch, show.**

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/mars-trace
echo "${MARSTRACE_API_KEY:?create one at console.marscompute.ai, MarsTrace keys: mt_live_...}"
marstrace hardware      # hardware ids, GPU, architecture, rates, agent model and limits
marstrace credit        # available credit and the page to add more
```

Every command has an MCP tool of the server `marstrace` (`marstrace mcp`, registered by
`.mcp.json`): `marstrace_hardware`, `marstrace_credit`, `marstrace_cubin_find`,
`marstrace_cubin_show`, `marstrace_estimate`, `marstrace_submit`, `marstrace_jobs`,
`marstrace_job`, `marstrace_wait`, `marstrace_cancel`, `marstrace_result`,
`marstrace_fetch`, `marstrace_show`. They take local paths and move the files themselves.
Use whichever is at hand; the steps below name both.

## 2. Settle the request

Settle three things with the user before writing anything:

- **The hardware.** Pick a hardware id from `marstrace hardware`: `h200`, `b200` or
  `b300`, among those it shows as available. A cubin runs only on the SM version it was
  built for: H200 is `sm_90a`, B200 `sm_100a`, B300 `sm_103a`.
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

[limits]
max_cost_usd = 40.0               # required: the most the job may cost
max_attempts = 2                  # optional, 1 to 3
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
- **Size the budget.** The agent's turns and tokens cap its session; `max_wall_seconds`
  bounds it in time, and the GPU is held for at most `max_wall_seconds + 120` seconds.
  `max_cost_usd` must cover the job fee, that GPU ceiling and at least $1 of agent tokens;
  what is left after the fee and the GPU ceiling is the agent's token budget.

## 5. Estimate, then submit

```bash
marstrace estimate my-job         # uploads, checks, quotes: cost cap, job fee, GPU ceiling, agent budget, token rates
```

**Tell the user the cost cap, the job fee, the GPU rate and the token rates, and get a yes
before the first submission of a session.** After that, keep them informed of the total.
A quote marked refused leaves the agent less than $1; raise `max_cost_usd` or lower
`max_wall_seconds`.

```bash
marstrace submit my-job --wait    # quotes again, asks, submits, waits, fetches
marstrace submit my-job --yes     # when the user already agreed to the cap
```

With MCP: `marstrace_estimate`, then `marstrace_submit` with `accept_max_cost_usd` set to
the cap the user agreed to (it refuses a higher quote), then `marstrace_wait` until
`status` is `succeeded`, `failed` or `cancelled`. Pass an `idempotency_key` (CLI:
`--idempotency-key`); when a submission's answer is lost, submit the same directory again
with the same key: the client resends the recorded request, and the service answers with
the job it already started.

- **Keep the cap near what the task needs** with `[limits] max_cost_usd` and the budget of
  `[agent]`, so the reservation on the user's credit stays small.
- **`insufficient_credit`, `spending_limit_exceeded` and `billing_suspended`** name the
  console page where credit is added; give it to the user. Do not retry.
- **`too_many_jobs`** means the workspace already runs its maximum of jobs; wait for one
  to end.
- **`marstrace cancel <job>`** stops a job the user no longer wants; once its GPU time
  started, it is charged for what it used.

Read the job's `error` when it fails (`marstrace job <id>`): `code` and `message`. A
failure the service caused is not charged; submit again and tell the user if it repeats.

## 6. Fetch and show

```bash
marstrace fetch <job>             # into marstrace-results/<job>/; --wait on submit does it
marstrace show marstrace-results/<job>/marstrace.json --stats
```

The result is `marstrace.json` and its `result-manifest.json`. Gate the trace before
reading conclusions from it (skill `marstrace-read`, section 2): the quality must be
`usable`, or each reason code explained. A session that ends without a trace has
`quality: "unusable"`; before submitting again, revisit the intent, the workspace and the
budget with the user. Results can be fetched for 30 days after the job ends.

## 7. Related skills

- `marstrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
