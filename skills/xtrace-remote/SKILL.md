---
name: xtrace-remote
description: Trace the timeline inside one CUDA kernel on a MarsCompute GPU with Xtrace, without a local GPU. Gather the kernel's cubins (from nvcc, Triton, torch.compile, TileLang or CuTeDSL output), and optionally its source, a launch description, input data and a driver script, into a workspace that an agent on an H200, B200 or B300 traces within a budget you set. Then estimate the cost, submit, fetch the trace and read it. Works through the `xtrace` CLI or its stdio MCP server (`xtrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel, on hardware they reach through MarsCompute.
---

# Xtrace on MarsCompute

Xtrace records when chosen parts of a compiled kernel execute. A job hands a workspace to
an agent on the GPU host: the cubins, and optionally the source tree, a launch manifest,
tensors and a driver script. The agent reads them, reproduces the launch, traces the
kernel within the budget you set, and returns one `xtrace.json` that `xtrace show` reads
in the terminal (skill `xtrace-read`).

The flow: **settle the request, find the cubin, gather the workspace, set the budget,
estimate, submit, fetch, show.**

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
echo "${MARSCOMPUTE_API_KEY:?create one at console.marscompute.ai, API keys: mc_live_v1_...}"
xtrace hardware      # environment ids, GPU, architecture, $/GPU-hour, agent models
xtrace credit        # available credit and the page to add more
```

Every command has an MCP tool of the server `xtrace` (`xtrace mcp`, registered by
`.mcp.json`): `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_estimate`, `xtrace_submit`, `xtrace_jobs`, `xtrace_job`,
`xtrace_wait`, `xtrace_cancel`, `xtrace_fetch`, `xtrace_show`. They take local paths and
move the files themselves. Use whichever is at hand; the steps below name both.

## 2. Settle the request

Settle three things with the user before writing anything:

- **The hardware.** Pick an environment id from `xtrace hardware`; it names the GPU and
  the CUDA environment the job runs in. A cubin runs only on the SM version it was built
  for: H200 is `sm_90a`, B200 `sm_100a`, B300 `sm_103a`. Every environment runs CUDA 12.9
  or later. When several environments of one GPU are listed, they differ in their CUDA
  environment, which the environment id names and `xtrace hardware --json` pins by
  `workspace_image_digest`, and in purchase option (`spot` or `on_demand`). Pick the one
  whose CUDA builds the user's source, and the cheaper rate when either does.
- **The kernel and the launch that matters.** A trace answers for the launch it observed:
  grid, block, problem size and data layout all change the timeline. Ask for the shape
  when the code leaves it open.
- **What to time.** Phases (load, compute, store, the wait on a barrier), warp roles, or
  per-iteration cost. Write it in `intent`, with the problem size; it is the agent's task.

## 3. Find the cubin

| Where the kernel comes from | How to get its cubin |
| --- | --- |
| CUDA C++ | `nvcc -cubin -arch=sm_90a -lineinfo -o k.cubin k.cu` (no GPU needed) |
| Triton, torch.compile | run it once, then `xtrace cubin find <name> --copy-to my-job` |
| TileLang | run it once, then `xtrace cubin find <name> --copy-to my-job` |
| CuTeDSL | have CuTeDSL keep its cubins in a directory (see its documentation on dumping compiled kernels), then `xtrace cubin find <name> --dir <that directory> --copy-to my-job` |
| only PTX at hand | `ptxas -arch=sm_90a -lineinfo -o k.cubin k.ptx`; the platform takes cubins only |

A cached cubin is built for the GPU the user ran it on. Check with
`xtrace cubin show my-job/<file>.cubin`; when its target differs from the hardware,
rebuild for that target. `sm_103a` needs CUDA 12.9 or later.

For Triton, `xtrace cubin find` prints what the cache recorded: the block size
(`num_warps × 32`), dynamic shared memory, cluster shape and the kernel's signature. Put
those facts in `intent` or a driver script, so the agent does not spend turns finding them.

## 4. Gather the workspace

```
my-job/
  xtrace.toml
  k.cubin                 the kernel, built for the hardware's architecture
  src/                    optional: the source tree; archived without .git
  launch.json             optional: a marscompute.kernel-launch manifest of the launch
  inputs.safetensors      optional: real input data
  run.py                  optional: a host driver script, a build file
```

```toml
hardware = "b200-cuda12.9"        # an environment id from xtrace hardware
intent = "find the longest bubble in the mainloop of the producer warps at M=N=K=8192"

[workspace]
cubins = ["k.cubin"]              # 1 to 8
source = "src"                    # optional: a directory, uploaded as source.tar.gz
manifest = "launch.json"          # optional
tensors = ["inputs.safetensors"]  # 0 to 8
files = ["run.py"]                # 0 to 8: driver scripts, build files

[agent]
model = "claude-opus-5-5"         # one of the agent models of xtrace hardware
max_turns = 200                   # 1 to 500
max_input_tokens = 20000000       # 1 to 50,000,000, cache reads and writes included
max_output_tokens = 200000        # 1 to 1,000,000
max_wall_seconds = 2400           # 60 to 2580: how long the agent works

[limits]
max_cost_usd = 40.0               # the most the job may cost
max_gpu_seconds = 3600
```

- **Give the agent a way to run the launch.** A driver script with the shapes the user
  cares about, a launch manifest, or source the agent can build; without one it derives
  the launch from source and spends turns on it. State the problem size in `intent`.
- **Real data only where it steers control flow.** Random inputs time a GEMM or an
  attention kernel as well as real ones. Send tensors for routing tables, block tables,
  sequence lengths and anything else the kernel branches on.
- **The workspace is shared on purpose.** The agent reads, builds and runs the source
  tree and the files. Leave out what the user does not want to share: secrets, data
  beyond the launch, unrelated code. `.git` is never uploaded.
- **Size the budget.** The agent's turns and tokens cap its session; `max_wall_seconds`
  bounds it in time. The GPU is held while the agent works, so set `max_gpu_seconds`
  above `max_wall_seconds`, and `max_cost_usd` above the job fee, the tokens at their
  caps and the GPU time. `xtrace estimate` prints the job fee and the token rates; a cap
  below their sum is refused.

## 5. Estimate, then submit

```bash
xtrace estimate my-job            # uploads, checks, quotes: cost cap, GPU rate, job fee, token rates
```

**Tell the user the cost cap, the GPU rate, the job fee and the token rates, and get a yes
before the first submission of a session.** After that, keep them informed of the total.

```bash
xtrace submit my-job --wait       # quotes again, asks, submits, waits, fetches
xtrace submit my-job --yes        # when the user already agreed to the cap
```

With MCP: `xtrace_estimate`, then `xtrace_submit` with `accept_max_cost_usd` set to the
cap the user agreed to (it refuses a higher quote), then `xtrace_wait` until `state` is
`succeeded`, `failed` or `cancelled`. Keep the `idempotency_key` of a submission whose
answer was lost and pass it again; it never starts a second job.

- **Lower the cap with `[limits]`** (`max_cost_usd`, `max_gpu_seconds`) and the budget of
  `[agent]` near what the task needs, so the hold on the user's credit stays small.
- **`insufficient_credit`** names the console page where credit is added; give it to the
  user. Do not retry.
- **`xtrace cancel <job>`** stops a job the user no longer wants; it settles at what it
  used.

Read the job's `error` when it fails (`xtrace job <id>`): `code`, `message`,
`classification` and `retryable`. A `platform_failure` is not the user's; submit again and
tell the user if it repeats.

## 6. Fetch and show

```bash
xtrace fetch <job>                # into xtrace-results/<job>/; --wait on submit does it
xtrace show xtrace-results/<job>/xtrace.json --stats
```

The result is `xtrace.json` and its `result-manifest.json`. Gate the trace before reading
conclusions from it (skill `xtrace-read`, section 2): the quality must be `usable`, or each
reason code explained. A session that ends without a trace has `quality: "unusable"`;
before submitting again, revisit the intent, the workspace and the budget with the user.

## 7. Related skills

- `xtrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
