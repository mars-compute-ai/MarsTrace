---
name: xtrace-remote
description: Trace the timeline inside one CUDA kernel on a MarsCompute GPU with Xtrace, without a local GPU or any tracing tool installed locally. Turn the user's request (which kernel, which phases, which GPU) into a job directory — the kernel's cubin or PTX (from nvcc, or from the Triton, torch.compile, TileLang or CuTeDSL cache), its launch shape and arguments, and an xtrace.toml — inspect the kernel's SASS to pick trace sites, submit the trace through the MarsCompute MCP server, and fetch the trace report. Use when the user wants per-phase durations, pipeline bubbles or warp-role scheduling inside a kernel, on hardware they reach through MarsCompute.
---

# Xtrace on MarsCompute

Xtrace records a timestamp, and optionally a register's value, each time chosen
instructions of a compiled kernel execute. The kernel is not rebuilt and its source is not
edited: the probes are placed into the SASS of the binary you send. MarsCompute loads that
binary on a GPU of the class you pick, launches the kernel as the job describes, and
returns a trace report that `xtrace show` reads in the terminal (skill `xtrace-read`).

No program of the user's runs on the platform. A job is data: the binary, the launch
shape, the arguments and the buffers' contents. An **inspect** job reads the binary and
returns the kernel's SASS listing and parameter layout, from which you pick sites; a
**trace** job launches the kernel once untraced and once traced, and returns the trace.

## 1. Check the setup

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
echo "${MARSCOMPUTE_API_KEY:?create one at console.marscompute.ai, API keys}"
xtrace hardware      # the GPU classes, their architecture and prices
```

The MCP server `marscompute` (`https://api.marscompute.ai/mcp`) offers the same calls as
tools: `xtrace_list_hardware`, `xtrace_get_balance`, `xtrace_submit_job`,
`xtrace_get_job`, `xtrace_list_jobs`, `xtrace_cancel_job`. Moving files is the CLI's
job: `xtrace upload` sends the bundle and `xtrace fetch` brings the results back. If the
MCP tools are missing, the CLI alone does everything (`xtrace submit --wait`).

## 2. Turn the request into a job

Settle three things with the user before writing anything:

- **The GPU.** Pick the hardware class from `xtrace hardware`. A cubin runs only on the
  SM version it was built for: an `sm_90a` cubin needs a Hopper class, an `sm_100a` one
  a B200 class, an `sm_103a` one a B300 class. Instrumentation supports `sm_90`, `sm_90a`,
  `sm_100`, `sm_100a`, `sm_103` and `sm_103a`.
- **The kernel and the launch that matters.** A trace answers for the launch it observed:
  grid, block, problem size and data layout all change the timeline. Ask for the shape
  when the code leaves it open.
- **What to time.** Phases (load, compute, store, the wait on a barrier), warp roles, or
  per-iteration cost. Write it in `intent`; it guides the sites you pick in section 3.

### 2.1 Get the binary

| Where the kernel comes from | How to get its binary |
| --- | --- |
| CUDA C++ | `nvcc -cubin -arch=sm_90a -lineinfo -o k.cubin k.cu` (no GPU needed) |
| Triton, torch.compile | run it once, then `xtrace cubin find <name> --copy-to my-job` |
| TileLang | run it once, then `xtrace cubin find <name> --copy-to my-job` |
| CuTeDSL | have CuTeDSL keep its cubins in a directory (see its documentation on dumping compiled kernels), then `xtrace cubin find <name> --dir <that directory> --copy-to my-job` |
| only PTX at hand | send the `.ptx`; the driver compiles it on the GPU |

The cached cubin is built for the GPU the user ran it on. Check with
`xtrace cubin show my-job/<file>.cubin`; when its target differs from the hardware
class, rebuild for that target (Triton and TileLang compile for a target without its GPU;
nvcc needs only `-arch`). Build with line info when you can, so sites can be named by
source line.

For Triton, `xtrace cubin find` prints what the cache recorded: the block size
(`num_warps × 32`), dynamic shared memory, cluster shape and the kernel's signature.
Triton passes its own scratch pointers after the listed arguments; the inspect job's
parameter layout (section 3) shows every parameter the binary takes.

### 2.2 Describe the launch

```
my-job/
  xtrace.toml          the manifest
  k.cubin              the binary
  k.cu                 optional: sources shown beside the listing
  inputs.safetensors   optional: buffer contents that must be real
```

```toml
hardware = "h100-sxm"
timeout_seconds = 120
intent = "per-iteration time of the main loop, producer warps against consumer warps"

[kernel]
binary = "k.cubin"
name = "square_matrix_mul"        # the entry point as the binary names it (mangled for C++)
sources = ["k.cu"]

[[buffers]]
name = "a"
dtype = "f16"                     # bool u8 i8 f8e4m3 f8e5m2 u16 i16 f16 bf16 u32 i32 f32 u64 i64 f64
shape = [4096, 4096]
init = "random"                   # zeros (default), ones, random, or file

[[buffers]]
name = "c"
dtype = "f32"
shape = [4096, 4096]
output = true                     # returned, and checked equal between the untraced and traced launch

[launch]
grid = [32, 32]
block = [128]
cluster = [2, 1]                  # optional
shared_bytes = 98304              # dynamic shared memory
args = [                          # in the order of the kernel's parameters
    { kind = "buffer", buffer = "a" },
    { kind = "buffer", buffer = "c", offset_bytes = 0 },
    { kind = "scalar", dtype = "i32", value = 4096 },
]
```

- **Buffers start random unless their values steer control flow.** Random data times a
  GEMM or an attention kernel as well as real data, and nothing of the user's leaves
  their machine. Use `init = "file"` with `tensors = "inputs.safetensors"` (a tensor of
  the buffer's name, dtype and shape) for routing tables, block tables, sequence lengths
  and anything else the kernel branches on.
- **A TMA descriptor** (`CUtensorMap` parameter) is described, not copied: the platform
  encodes it over a buffer with `cuTensorMapEncodeTiled`'s parameters:
  `{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [K, M],
  global_strides_bytes = [K * 2], box_dims = [64, 128], swizzle = "128b" }`; read the
  values from the code that builds it on the host.
- **A by-value struct** (a CUTLASS `Params`) is `{ kind = "bytes", hex = "...",
  relocations = [{ offset = 16, buffer = "a" }] }`: its bytes, with the offsets where a
  buffer's device address goes.
- **Check the parameter layout before tracing.** Run the inspect job first; its parameter
  sizes must match your `args` one for one.

The bundle holds the manifest and the files it names, nothing else, at most 512 MiB.

## 3. Inspect, then pick sites

```bash
xtrace submit my-job --kind inspect --wait
xtrace sass my-job/xtrace-results/<job>/inspect.json --kernels
xtrace sass my-job/xtrace-results/<job>/inspect.json --line matmul.cu:9
xtrace sass my-job/xtrace-results/<job>/inspect.json --grep 'HGMMA|UTMALDG|SYNCS' --first 200 --last 320
```

`--kernels` lists the binary's kernels, the named one starred, and the named kernel's
registers, static shared memory and parameters (index, offset, size). Compare the
parameters with `launch.args` before you trace.

One source line maps to many instructions across several basic blocks. Enumerate them;
do not take the first one blindly.

- **A site records each time its instruction executes.** A loop-body site writes one
  record per iteration, a site before the loop writes one. Know which you want.
- **Prefer an instruction followed by plenty of the kernel's own work in its basic
  block.** The probe's store then overlaps that work instead of stalling in front of it.
- **Never put a site inside a spin loop** (a barrier or mbarrier poll). Every turn writes
  a record and floods the ring. Bracket the wait from outside; it shows up as the gap
  between intervals.
- **Pair sites into regions.** A region is an `enter` site and a `leave` site that the
  same threads execute equally often, `enter` first. Regions are what the report turns
  into intervals, statistics and bubbles. Give each region its own two sites: a site
  that leaves one region and enters the next is paired as an enter only.
- **Split waiting from working.** A role that parks on a barrier and then works gets
  one region for the wait and one for the work.
- **Budget the ring.** Each thread keeps its last 32 records by default
  (`per_thread_records`). Keep twice the executions of every region well under it; a
  count sitting exactly at the depth means the ring wrapped.

For a warp-specialized kernel (producer, consumer and scheduler warps), map which warps
run which blocks before choosing regions: trace a few candidate sites alone, then read
`xtrace show --stats` per `--warp`. The CFG says which paths exist, not which warps take
them.

```toml
[trace]
clock = "gpu"                    # gpu: ns, comparable across SMs; sm: cycles; anchor: ns, warp-level only

[[trace.sites]]                  # site 0
line = "matmul.cu:8"

[[trace.sites]]                  # site 1
index = 188                      # by instruction index, from xtrace sass
value_register = "R12"           # optional: record a register beside the timestamp

[[trace.regions]]
enter = 0
leave = 1
label = "dot product"
```

`warp_level = true` on every site records once per warp instead of once per thread. Use
it for warps that run short of registers, or to keep a longer timeline in the ring. A run
mixes no thread-level and warp-level sites.

## 4. Submit the trace

With MCP:

```bash
xtrace upload my-job          # prints {upload_id, hardware, spec}
```

then call `xtrace_submit_job` with exactly those three fields, poll `xtrace_get_job`
every half minute or so until `state` is final, and run `xtrace fetch <job>`.

Without MCP:

```bash
xtrace submit my-job --wait   # results land in my-job/xtrace-results/<job>/
```

Read the job's `error` when it fails:

| `error.code` | What to do |
| --- | --- |
| `binary_invalid` | the binary does not load on the class's GPU; check its target with `xtrace cubin show` |
| `kernel_not_found` | the binary has no such entry point; `xtrace sass --kernels` on an inspect result |
| `arch_unsupported` | the class's architecture cannot be instrumented |
| `launch_invalid` | `args` do not match the kernel's parameters, or a buffer or tensor map cannot be built; `log.txt` says which |
| `launch_failed` | the untraced launch failed (an illegal address, a bad launch shape); fix the launch, it is charged |
| `site_unresolved` | a `line` has no instruction; pick an index from the listing |
| `site_not_instrumentable` | the code around that site has no registers to spare: move the site one or two instructions, or make it `warp_level`; say which you chose |
| `timeout` | the job ran past `timeout_seconds` |
| `platform_failure` | not charged; submit again, and tell the user if it repeats |

Never fix a refused site by silently dropping it. A region that could not be probed is a
result; report it.

## 5. Gate the trace before reading conclusions from it

Run these on every trace (`xtrace show` views from skill `xtrace-read`):

- **Pair rate.** `xtrace show trace.json --stats` lists each region's pair rate. Below
  100% means boundaries went unpaired: a wrapped ring, a region whose sites run unequal
  counts, or a rare corrupted record. Say which before using the region's numbers.
- **Records per thread match your model.** A loop of N iterations gives N records per
  thread at a loop-body site (unless the ring wrapped). A different count means the site
  is not where you thought.
- **Every declared site appears.** A missing site never executed on the traced path,
  under this shape.
- **Spans make sense.** An interval close to the whole kernel is a wait you did not split
  out. A leave before its enter means the region is backwards.
- **Outputs match.** `summary.json` carries `outputs_match` for the buffers marked
  `output`. `false` means the traced launch computed something else; do not report its
  timeline until the cause is known.
- **Read `log.txt`.** Warnings such as warps reaching their record budget mean the
  counts are lower bounds.

## 6. Cost

A trace job costs a fixed price plus a price per GPU-second, as `xtrace hardware` lists;
an inspect job costs the fixed price only. Submission reserves the price of the whole
`timeout_seconds` against the user's credit and refuses the job when the credit does not
cover it (`xtrace_get_balance`); the error names the console page where credit is added.
Failed launches and timeouts are charged for the time they ran; platform failures are
not.

- **Tell the user what a job will cost at most, and get a yes, before the first
  submission of a session.** After that, keep them informed of the total.
- **Set `timeout_seconds` near the real run time** (a few times the local run time), so
  the reservation stays small and a hung kernel stops early.
- **Inspect once per kernel build.** Instruction indices hold for one cubin only; after a
  rebuild, inspect again or name sites by line.

## 7. Related skills

- `xtrace-read`: read the trace report in the terminal and find the bubbles.
