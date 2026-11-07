---
name: xtrace-remote
description: Trace the timeline inside one CUDA kernel on a MarsCompute GPU with Xtrace, without a local GPU. Find the kernel's cubin (nvcc, or the Triton, torch.compile, TileLang or CuTeDSL cache), describe its launch in an xtrace.toml, inspect its SASS to pick trace sites, estimate the cost, submit, fetch the results and read them. Works through the `xtrace` CLI or its stdio MCP server (`xtrace mcp`). Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel, on hardware they reach through MarsCompute.
---

# Xtrace on MarsCompute

Xtrace records a timestamp, and optionally a register's value, each time chosen
instructions of a compiled kernel execute. The kernel is not rebuilt and its source is not
edited. MarsCompute loads the cubin you send on the hardware you pick, launches the kernel
as the job describes, and returns a trace report that `xtrace show` reads in the terminal
(skill `xtrace-read`).

No program of the user's runs on the platform. A job is data: the cubin, the launch
shape, the arguments and the buffers' contents.

The flow: **find the cubin, inspect, pick sites, estimate, submit, fetch, show.**

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
echo "${MARSCOMPUTE_API_KEY:?create one at console.marscompute.ai, API keys: mc_live_v1_...}"
xtrace hardware      # environment ids, GPU, architecture, $/GPU-hour, stages
xtrace credit        # available credit and the page to add more
```

Every command has an MCP tool of the server `xtrace` (`xtrace mcp`, registered by
`.mcp.json`): `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_inspect`, `xtrace_sass`, `xtrace_estimate`, `xtrace_submit`,
`xtrace_jobs`, `xtrace_job`, `xtrace_wait`, `xtrace_cancel`, `xtrace_fetch`,
`xtrace_show`. They take local paths and move the files themselves. Use whichever is at
hand; the steps below name both.

## 2. Settle the request

Settle three things with the user before writing anything:

- **The hardware.** Pick an environment id from `xtrace hardware`. A cubin runs only on
  the SM version it was built for: `sm_90a` needs a Hopper environment, `sm_100a` a
  B200 one, `sm_103a` a B300 one. Check that the stages you want (`xtrace`,
  `range_profile`, `pc_sampling`) are listed for it.
- **The kernel and the launch that matters.** A trace answers for the launch it observed:
  grid, block, problem size and data layout all change the timeline. Ask for the shape
  when the code leaves it open.
- **What to time.** Phases (load, compute, store, the wait on a barrier), warp roles, or
  per-iteration cost. Write it in `intent`; it is the job's prompt.

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
rebuild for that target. Build with line info when you can, so sites can be named by
source line.

For Triton, `xtrace cubin find` prints what the cache recorded: the block size
(`num_warps × 32`), dynamic shared memory, cluster shape and the kernel's signature.
Triton passes its own scratch pointers after the listed arguments; the inspection's
parameter layout (section 5) shows every parameter the cubin takes.

## 4. Describe the launch

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
name = "a"
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
output = true                     # arguments naming it may write it

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
  their machine. Use `init = "tensor"` with `tensors` for routing tables, block tables,
  sequence lengths and anything else the kernel branches on.
- **A TMA descriptor** (`CUtensorMap` parameter) is described, not copied:
  `{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [K, M],
  global_strides_bytes = [K * 2], box_dims = [64, 128], swizzle = "128b" }`, with
  `cuTensorMapEncodeTiled`'s parameters, innermost dimension first. Read the values from
  the code that builds it on the host.
- **A by-value struct** (a CUTLASS `Params`) is `{ kind = "bytes", hex = "...",
  relocations = [{ offset = 16, buffer = "a" }] }`: its bytes, with the 8-aligned offsets
  where a buffer's device address goes.
- **Check the result against a reference** when correctness matters:
  `reference = "reference.safetensors"` and `[validation]` with `output = "c"`, `atol`,
  `rtol`. The reference file holds a tensor of the output buffer's name, type and shape.

## 5. Inspect, then pick sites

Inspections are free and cached beside the cubin, keyed by its digest.

```bash
xtrace inspect my-job/k.cubin --kernel gemm_kernel     # kernels, registers, parameters
xtrace sass my-job/k.cubin --kernel gemm_kernel --line k.cu:9
xtrace sass my-job/k.cubin --kernel gemm_kernel --grep 'HGMMA|UTMALDG|SYNCS' --first 200 --last 320
```

Compare the parameters (index, offset, size) with `launch.args` one for one; `estimate`
refuses a mismatch when a listing is cached. `+` marks the instructions that take a
probe; a site on any other instruction is refused.

One source line maps to many instructions across several basic blocks. Enumerate them;
do not take the first one blindly. A site named by `line` resolves to the first `+`
instruction of that line.

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
line = "k.cu:9"

[[trace.sites]]                  # site 1
index = 188                      # an instruction index from xtrace sass
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
`max_input_tokens`, `max_output_tokens`.

## 6. Estimate, then submit

```bash
xtrace estimate my-job            # uploads, resolves the sites, quotes: cost cap and rate
```

**Tell the user the cost cap and the rate, and get a yes, before the first submission of
a session.** After that, keep them informed of the total.

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
- **Inspect once per kernel build.** Instruction indices hold for one cubin only; after a
  rebuild, inspect again or name sites by line.

Read the job's `error` when it fails (`xtrace job <id>`): `code`, `message`,
`classification` and `retryable`. A `platform_failure` is not the user's; submit again
and tell the user if it repeats. Never fix a refused site by silently dropping it; a
region that could not be probed is a result, so report it.

## 7. Fetch and show

```bash
xtrace fetch <job>                # into xtrace-results/<job>/; --wait on submit does it
xtrace show xtrace-results/<job>/xtrace.json --stats
```

Gate the trace before reading conclusions from it (skill `xtrace-read`, section 2):
`summary.json` must say `quality: usable`, or explain each reason code, and
`correctness.json` must pass.

## 8. Related skills

- `xtrace-read`: read the trace report in the terminal, gate it, and find the bubbles.
