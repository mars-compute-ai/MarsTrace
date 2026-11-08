---
name: xtrace-remote
description: Trace the timeline inside one CUDA kernel on a MarsCompute GPU with Xtrace, without a local GPU. Either describe one launch of a cubin in an xtrace.toml (find the cubin in nvcc, Triton, torch.compile, TileLang or CuTeDSL output, run a paid inspection to read its SASS, pick trace sites), or hand an agent on an H200, B200 or B300 a workspace of cubins, source and data in agent mode. Then estimate the cost, submit, fetch the results and read them. Works through the `xtrace` CLI or its stdio MCP server (`xtrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel, on hardware they reach through MarsCompute.
---

# Xtrace on MarsCompute

Xtrace records a timestamp, and optionally a register's value, each time chosen
instructions of a compiled kernel execute. MarsCompute runs the kernel on the hardware you
pick and returns a trace report that `xtrace show` reads in the terminal (skill
`xtrace-read`).

A job takes one of two forms:

- **A launch job** is data: the cubin, the launch shape, the arguments and the buffers'
  contents, and the sites to trace. You pick the sites from a paid inspection of the
  cubin. The flow: **find the cubin, inspect, pick sites, estimate, submit, fetch, show.**
- **An agent-mode job** hands a workspace to an agent on the GPU host: cubins, and
  optionally the source tree, a launch manifest, tensors and a driver script. The agent
  reads them, builds and runs what it needs within the budget you set, and returns one
  `xtrace.json`. The flow: **gather the workspace, set the budget, estimate, submit,
  fetch, show.**

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
echo "${MARSCOMPUTE_API_KEY:?create one at console.marscompute.ai, API keys: mc_live_v1_...}"
xtrace hardware      # environment ids, GPU, architecture, $/GPU-hour, stages, bundle kinds, models
xtrace credit        # available credit and the page to add more
```

Every command has an MCP tool of the server `xtrace` (`xtrace mcp`, registered by
`.mcp.json`): `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_inspection_capabilities`, `xtrace_inspection_estimate`,
`xtrace_inspection_submit`, `xtrace_inspection_get`, `xtrace_inspection_wait`,
`xtrace_inspection_query`, `xtrace_inspection_cancel`, `xtrace_inspection_result`,
`xtrace_inspection_downloads`, `xtrace_sass`, `xtrace_estimate`, `xtrace_submit`,
`xtrace_jobs`, `xtrace_job`, `xtrace_wait`, `xtrace_cancel`, `xtrace_fetch`,
`xtrace_show`. They take local paths and move the files themselves. Use whichever is at
hand; the steps below name both.

## 2. Settle the request

Settle four things with the user before writing anything:

- **The form of the job.** Use a launch job when the user can say which instructions to
  time and can describe the launch: grid, block, arguments, buffers. Use agent mode when
  the launch is easiest to reproduce from the user's own code (a host script, a build, a
  framework call), or when the user cannot say what to trace.
- **The hardware.** Pick an environment id from `xtrace hardware`; it names the GPU and
  the CUDA environment the job runs in. A cubin runs only on the SM version it was built
  for: `sm_90a` needs a Hopper environment (H100, H200), `sm_100a` a B200 one, `sm_103a` a
  B300 one. A launch job needs `cubin_launch` among the environment's bundles; agent mode
  needs `agent_workspace`. Check that the stages you want
  (`xtrace`, `range_profile`, `pc_sampling`) are listed for a launch job.
- **The kernel and the launch that matters.** A trace answers for the launch it observed:
  grid, block, problem size and data layout all change the timeline. Ask for the shape
  when the code leaves it open.
- **What to time.** Phases (load, compute, store, the wait on a barrier), warp roles, or
  per-iteration cost. Write it in `intent`; it is the job's prompt, and in agent mode the
  agent's task.

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
(`num_warps × 32`), dynamic shared memory, cluster shape and the kernel's signature.
Triton passes its own scratch pointers after the listed arguments; the inspection's
parameter layout (section 5) shows every parameter the cubin takes.

## 4. Describe the launch

A launch job's directory:

```
my-job/
  xtrace.toml             the manifest
  k.cubin                 the cubin
  k.cu                    optional: sources, kept local for the viewer
  inputs.safetensors      optional: buffer contents that must be real
  reference.safetensors   optional: the expected output, with [validation]
```

```toml
hardware = "h100-sxm"             # an environment id from xtrace hardware
intent = "per-iteration time of the main loop, producer warps against consumer warps"
tensors = "inputs.safetensors"    # only with an init = "tensor" buffer

[kernel]
binary = "k.cubin"
name = "gemm_kernel"              # the entry point as the cubin names it (mangled for C++)
sources = ["k.cu"]

[[buffers]]
name = "a"                        # letters, digits and _, at most 128
dtype = "f16"                     # bool u8 i8 f8e4m3 f8e5m2 u16 i16 f16 bf16 u32 i32 f32 u64 i64 f64
shape = [4096, 4096]
init = "random"                   # zeros (default), ones, random, tensor
seed = 1

[[buffers]]
name = "routes"
dtype = "i32"
shape = [4096]
init = "tensor"                   # the tensor "routes" of the tensors file

[[buffers]]
name = "c"
dtype = "f32"
shape = [4096, 4096]
output = true                     # the kernel writes it; the run is checked on it

[launch]
grid = [32, 32]
block = [128]
cluster = [2, 1]                  # optional; at most 16 blocks
shared_bytes = 98304              # dynamic shared memory
args = [                          # in the order of the kernel's parameters
    { kind = "buffer", buffer = "a" },
    { kind = "buffer", buffer = "routes" },
    { kind = "buffer", buffer = "c", offset_bytes = 0 },   # a multiple of 16
    { kind = "scalar", dtype = "i32", value = 4096 },
]

[limits]
max_cost_usd = 2.0                # the most the job may cost
max_gpu_seconds = 120
```

- **Buffers start random unless their values steer control flow.** Random data times a
  GEMM or an attention kernel as well as real data, and nothing of the user's leaves
  their machine. The harness prepares `ones` and `random` buffers itself and uploads them
  with the tensors, so a large one costs its full size in upload. Use `init = "tensor"`
  with `tensors` for routing tables, block tables, sequence lengths and anything else the
  kernel branches on.
- **Mark every buffer the kernel writes as `output = true`**, including scratch and
  workspace buffers; the launch restores every buffer before each run and checks the
  outputs. An `f8` output is compared byte for byte and cannot be the `[validation]`
  output.
- **A TMA descriptor** (`CUtensorMap` parameter) is described, not copied:
  `{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [K, M],
  global_strides_bytes = [K * 2], box_dims = [64, 128], swizzle = "128b" }`, with
  `cuTensorMapEncodeTiled`'s parameters, innermost dimension first. Read the values from
  the code that builds it on the host; the map must fit inside its buffer.
- **A by-value struct** (a CUTLASS `Params`) is `{ kind = "bytes", hex = "...",
  relocations = [{ offset = 16, buffer = "a" }] }`: its bytes, with zeros at the 8-aligned
  offsets where a buffer's device address goes. The arguments together hold at most 4096
  bytes.
- **Check the result against a reference** when correctness matters:
  `reference = "reference.safetensors"` and `[validation]` with `output = "c"`, `atol`,
  `rtol`. The reference file holds a tensor of the output buffer's name, type and shape.

## 5. Inspect, then pick sites

An inspection reads the cubin's kernels, their parameter layout and the SASS listing of
one kernel. It is quoted and charged a fixed fee. **Tell the user the fee and get a yes
before the first inspection of a session**, as for a job.

```bash
xtrace inspection profiles                          # the profiles and their fees
xtrace inspect my-job/k.cubin --kernel gemm_kernel  # quotes, asks, runs, caches the listing
xtrace sass my-job/k.cubin --kernel gemm_kernel --grep 'HGMMA|UTMALDG|SYNCS' --first 200 --last 320
```

With MCP: `xtrace_inspection_estimate` for the fee, `xtrace_inspection_submit` with
`accept_max_fee_usd` set to the fee the user agreed to (it refuses a higher quote),
`xtrace_inspection_wait` until `execution.state` is `terminal`, then
`xtrace_inspection_result` with the cubin and the kernel, which caches the listing beside
the cubin for `xtrace_sass`. The listing is cached by the cubin's digest: inspect once per
build, and keep the `idempotency_key` of an acceptance whose answer was lost and pass it
again. `xtrace inspection get <id>` and `xtrace inspection result <id> <cubin> --kernel
<name>` pick up an inspection that was left running.

Compare the parameters (index, offset, size) with `launch.args` one for one; `estimate`
refuses a mismatch when an inspection is cached, and the launch manifest carries the
layout. An outcome other than `completed` (`rejected_input` for a cubin the profile cannot
read) is the user's to see; do not retry it unchanged.

The listing prints each instruction's index, its address and its SASS. A site names its
instruction by index. The listing carries no source lines; to find the instructions of a
source line, disassemble the cubin locally with line info (`nvdisasm -c -g k.cubin`, no
GPU needed) and match by address, which is the same in both. One source line maps to many
instructions across several basic blocks; enumerate them, do not take the first one
blindly.

- **A site records each time its instruction executes.** A loop-body site writes one
  record per iteration, a site before the loop writes one. Know which you want.
- **Never put a site inside a spin loop** (a barrier or mbarrier poll). Every turn writes
  a record and floods the ring. Bracket the wait from outside; it shows up as the gap
  between intervals.
- **Pair sites into regions.** A region is an `enter` site and a `leave` site that the
  same threads execute equally often, `enter` first. Give each region its own two sites:
  a site that leaves one region and enters the next is paired as an enter only.
- **Split waiting from working.** A role that parks on a barrier and then works gets
  one region for the wait and one for the work.
- **Budget the ring.** Each thread keeps its last `per_thread_records` records (32 by
  default). Keep twice the executions of every region well under it; a count at exactly
  the depth means the ring wrapped (`record_capacity_reached_possible_wrap`).

For a warp-specialized kernel, map which warps run which blocks before choosing regions:
trace a few candidate sites alone, then read `xtrace show --stats` per `--warp`.

```toml
[trace]
clock = "gpu"                    # gpu: ns across SMs; sm: cycles; anchor: ns, warp-level only
per_thread_records = 64

[[trace.sites]]                  # site 0
index = 141                      # an instruction index from xtrace sass

[[trace.sites]]                  # site 1
index = 188
value_register = "R12"           # optional: also record a register (R0-R254, UR0-UR62)

[[trace.regions]]
enter = 0
leave = 1
label = "mainloop"               # letters, digits, _ . : -

[trace.scope_roles]
mainloop = "consumer"            # optional: the warp role of a region, for the report
```

`warp_level = true` on every site records once per warp. Use it for warps that run short
of registers, or to keep a longer timeline. `value_is_smid = true` records the SM a block
runs on. `stages = ["xtrace", "range_profile"]` with `[trace.range_profile]` (`metrics`,
`max_invocations`) or `[trace.pc_sampling]` (`repeats`) adds a profiler stage.

When the user cannot say what to trace, `[planner]` instead of `[trace]` has the platform
pick the sites from the intent: `model` (from `xtrace hardware`), `max_calls`,
`max_input_tokens`, `max_output_tokens`. Agent mode (section 6) goes further: the agent
also prepares the launch itself.

## 6. Agent mode

An agent-mode job needs no inspection and no `[kernel]`, `[launch]`, `[[buffers]]` or
`[trace]`. Gather what the agent needs to reproduce the launch the user cares about:

```
my-agent-job/
  xtrace.toml
  k.cubin                 the kernel, built for the hardware's architecture
  src/                    optional: the source tree; archived without .git
  launch.json             optional: a marscompute.kernel-launch manifest of the launch
  inputs.safetensors      optional: real input data
  run.py                  optional: a host driver script, a build file
```

```toml
hardware = "b200-cuda12.9"        # an environment whose bundles list agent_workspace
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
max_cost_usd = 40.0
max_gpu_seconds = 3600
```

- **Give the agent a way to run the launch.** A launch manifest, or a driver script with
  the shapes the user cares about, or source the agent can build; without one it derives
  the launch from source and spends turns on it. State the problem size in `intent`.
- **The workspace is shared on purpose.** The agent reads, builds and runs the source
  tree and the files. Leave out what the user does not want to share: secrets, data
  beyond the launch, unrelated code. `.git` is never uploaded.
- **Pick the hardware by its GPU and environment.** H200 is `sm_90a`, B200 `sm_100a`, B300
  `sm_103a`; every cubin must target the environment's architecture. Agent-mode
  environments run CUDA 12.9 or later. When `xtrace hardware` lists several environments
  of one GPU, they differ in their CUDA environment, which the environment id identifies and
  `xtrace hardware --json` pins by `image_digest`, and in purchase option (`spot` or
  `on_demand`). Pick the environment whose CUDA builds the user's source, and the cheaper
  rate when either does.
- **Size the budget.** The agent's turns and tokens cap its session; `max_wall_seconds`
  bounds it in time. The GPU is held while the agent works, so set `max_gpu_seconds`
  above `max_wall_seconds`, and `max_cost_usd` above the job fee, the tokens at their caps
  and the GPU time. `xtrace estimate` prints the job fee and the token rates; a cap below
  their sum is refused.

The result is one `xtrace.json` and its `result-manifest.json`; read it with `xtrace show`
(skill `xtrace-read`). A session that ends without a trace has `quality: "unusable"`.

## 7. Estimate, then submit

```bash
xtrace estimate my-job            # uploads, checks, quotes: cost cap, rate, and fees in agent mode
```

**Tell the user the cost cap and the rate, and in agent mode the job fee and the token
rates, and get a yes before the first submission of a session.** After that, keep them
informed of the total.

```bash
xtrace submit my-job --wait       # quotes again, asks, submits, waits, fetches
xtrace submit my-job --yes        # when the user already agreed to the cap
```

With MCP: `xtrace_estimate`, then `xtrace_submit` with `accept_max_cost_usd` set to the
cap the user agreed to (it refuses a higher quote), then `xtrace_wait` until `state` is
final. Keep the `idempotency_key` of a submission whose answer was lost and pass it again;
it never starts a second job.

- **Lower the cap with `[limits]`** (`max_cost_usd`, `max_gpu_seconds`) near the real
  run time, so the hold on the user's credit stays small and a hung kernel stops early.
- **`insufficient_credit`** names the console page where credit is added; give it to the
  user. Do not retry.
- **Instruction indices hold for one cubin only.** After a rebuild, inspect again and
  pick the sites anew.

Read the job's `error` when it fails (`xtrace job <id>`): `code`, `message`,
`classification` and `retryable`. A `platform_failure` is not the user's; submit again
and tell the user if it repeats. Never fix a refused site by silently dropping it; a
region that could not be probed is a result, so report it.

## 8. Fetch and show

```bash
xtrace fetch <job>                # into xtrace-results/<job>/; --wait on submit does it
xtrace show xtrace-results/<job>/xtrace.json --stats
```

Gate the trace before reading conclusions from it (skill `xtrace-read`, section 2): the
quality must be `usable`, or each reason code explained, and the correctness checks must
pass.

## 9. Related skills

- `xtrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
