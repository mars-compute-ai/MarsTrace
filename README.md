# MarsTrace

MarsTrace times the inside of a CUDA kernel on a cloud GPU. You send the kernel's cubin, and
an agent on a GPU machine instruments it, runs it at the launch you describe and returns
`marstrace.json`: per-region durations for every thread or warp, the balance of the grid, and
an analysis of where the time goes. This repository is the source-available client, the
`marstrace` command and its MCP server. It is for kernel developers and their coding agents, and it needs no
local GPU.

[中文版](README.zh.md)

## Quick start

1. Register at the [MarsCompute console](https://console.marscompute.ai/register), add
   credit, and create an API key under MarsTrace › Keys.
2. Install the client and trace the bundled example:

```bash
pip install 'git+https://github.com/mars-compute-ai/MarsTrace@v0.3.0'
export MARSTRACE_API_KEY=mt_live_...

marstrace estimate examples/matmul        # token rates, the most the job can cost, your balance
marstrace submit examples/matmul --wait   # submits, waits, downloads, prints the analysis
marstrace show examples/matmul/marstrace-results/<job>/marstrace.json --stats
```

A job directory holds a `marstrace.toml` and the files it names:

```toml
hardware = "h200"   # or "b200"; build the cubin for its architecture (sm_90a or sm_100a)
intent = """
How long each thread spends in the dot-product loop of square_matrix_mul, and how evenly the
blocks run, at n = 256: int32 matrices a, b and c of 256 x 256, a and b random, launched with
grid (16, 16) and block (16, 16).
"""

[workspace]
cubins = ["matmul.cubin"]   # nvcc -cubin -arch=sm_90a -lineinfo -o matmul.cubin src/matmul.cu
source = "src"              # optional; it lets the agent name source lines
```

Write in `intent` the question, the problem size and the launch's grid and block. The agent
traces exactly that launch. For a Triton, CuTeDSL or TileLang kernel, `marstrace cubin find
<kernel> --copy-to my-job` copies its cubin out of the compiler's cache.
[docs/protocol.md](docs/protocol.md) lists every key of `marstrace.toml`.

## What it answers

MarsTrace records a timestamp each time a thread or warp reaches an instruction you care
about, so it measures time between points inside the kernel:

- how long each loop, phase or warp role takes, per thread or per warp, with its spread;
- whether blocks and SMs are evenly loaded, how many blocks run on an SM at once, and the
  tail at the end of the grid;
- where producer and consumer warps wait on each other, and the gaps between regions.

It does not measure DRAM or L2 traffic, coalescing, stall reasons, bank conflicts or
tensor-pipe use. The analysis names the profiler metric that would settle each of those.
A fast job does not change your kernel; the thinking modes below do.

## Reading the result

`marstrace fetch` and `submit --wait` save two files in `marstrace-results/<job>/`:

- `result-manifest.json`: read `quality` first. `usable` means every check passed;
  `partial` comes with `reason_codes` that say which numbers to distrust; `unusable` means
  no timeline, and such a job is not charged.
- `marstrace.json`: the trace, which `marstrace show` reads, and the analysis, which
  `marstrace fetch` prints and `marstrace show <file> --analysis` prints again.

The analysis comes from a deterministic analyser; no model writes it. It holds:

- **Findings**, each with its evidence from this run, a next step, and **Where**: the region
  and each boundary site's SASS index, opcode and source line.
- **Hypotheses** under a finding, and **what this trace cannot tell**: causes a timing trace
  cannot see, each with the measurement that would confirm it. A hypothesis is not a cause
  until that measurement is made.
- **Balance**: SM utilization, the tail, blocks resident per SM, the traced span and the
  kernel time without instrumentation.

## Optimizing a kernel

The modes `thinking-light` and `thinking-heavy` trace the kernel as a fast job does, then
let the agent change its source, one measured iteration at a time, and keep only the changes
that make it faster and still pass your check. The job directory adds the source tree and an
`[optimize]` table with the commands the agent runs in its root:

```toml
mode = "thinking-light"

[workspace]
cubins = ["matmul.cubin"]
source = "src"                  # required: the tree the agent builds and changes

[optimize]
build = "./build.sh"            # optional
run = "python3 run.py"          # required: launches the kernel at the workload that matters
verify = "python3 verify.py"    # required: exits 0 when the results are correct
kernel = "square_matrix_mul"    # optional: a regular expression of the target kernel's name
```

```bash
marstrace estimate examples/matmul-optimize
marstrace submit examples/matmul-optimize --wait
```

The build and the run have no network access, so every dependency the image lacks must ship
in the source archive. `marstrace job` shows the iteration, the changes kept and reverted,
the best speedup so far and the time spent against the budget. `fetch` writes
`optimized.patch`, the performance, code-change and process reports, and each iteration's
files under `marstrace-results/<job>/optimization/`, and prints each configuration's time
before and after. `marstrace show <file> --optimization` prints it again.
`thinking-heavy` has a larger budget than `thinking-light`; the caps of each are in
[docs/protocol.md](docs/protocol.md).

## Agents: Claude Code plugin and MCP

The plugin adds the `marstrace mcp` server and two skills, `marstrace-remote` (prepare and
submit a job) and `marstrace-read` (read the result):

```bash
claude plugin marketplace add mars-compute-ai/MarsTrace
claude plugin install marstrace@marscompute
```

Other agents run `marstrace mcp` as a stdio MCP server with `MARSTRACE_API_KEY` in its
environment, as [.mcp.json](.mcp.json) does, and load the skills from
[marstrace/skills/](marstrace/skills/).

## What is supported now

MarsTrace is in early access: jobs run once MarsCompute enables your workspace. Open an
issue or contact MarsCompute to ask for access.

- Mode `fast`: a trace and its analysis.
- Modes `thinking-light` and `thinking-heavy`: a trace, then an optimized kernel with its patch
  and reports; `thinking-heavy` runs up to five hours. A run may end earlier when it has
  measured and ruled out every candidate change; the process report lists them, and you pay
  only for the tokens it used.
- Very large persistent kernels (a launch of thousands of blocks of which one per SM runs, such
  as FlashAttention 4): per-iteration timings may come out `partial`, because a trace keeps
  100,000 records; per-tile timings are delivered.
- GPUs: H200 and B200. `marstrace hardware` lists the hardware your account can use.
- CUDA 13.0.
- Inputs: up to 8 cubins (ELF cubins, not PTX), plus optional source, a launch manifest,
  safetensors inputs and other files, 256 MiB in all.
- Rate limits per API key: 60 requests, 30 upload grants and 5 submissions a minute, and 2
  running and 10 queued jobs per workspace. The client waits and retries a rate-limited
  request on its own.

## Billing

A job is charged the agent's tokens over all its attempts, at the rates `marstrace hardware`
and `marstrace estimate` show. GPU time is not charged separately. A job whose trace is
`unusable`, or that fails without delivering a result, is not charged. A cancelled job pays
for the tokens used up to the stop. A job starts only while your balance reaches the
minimum of its mode: $100 for `fast` and `thinking-light`, and $300 for `thinking-heavy`.
`marstrace credit` shows the current minimum.

`marstrace estimate` shows two figures. **Typical** is the middle half of what recent jobs of
the same mode were charged, a guide rather than a bound. **At most** is a ceiling the service
enforces. In the thinking modes each attempt stops at its cost cap, $250 for
`thinking-light` and $1,000 for `thinking-heavy` by default, so a default job costs at most
$500 or $3,000. Set `max_attempt_cost_usd` in `[limits]` to change the cap. A fast job's
ceiling prices every input token at the dearest input rate, well above what its mostly cached
calls cost.

## Privacy and security

- Each attempt runs on its own GPU machine, which is deleted when the attempt ends. Nothing
  is shared between jobs or customers.
- The machine accepts no inbound connection, and the agent's container reaches nothing but
  the MarsTrace service.
- Your uploaded files are deleted 24 hours after the job ends. Results can be downloaded
  for 30 days.
- MarsCompute keeps the agent's model requests and replies for each job, which may quote
  your code, for 30 days after the job ends, and then deletes them. Only MarsCompute
  administrators can read them, and they are deleted earlier on your request.

## FAQ

**Do I need a GPU or CUDA locally?** No. You need `nvcc`, or a compiler cache, only to
build the cubin.

**Which kernels work?** Any kernel you can hand over as a cubin built for the GPU's
architecture: CUDA C++, Triton, CuTeDSL, TileLang, CUTLASS.

**What if my kernel needs a special launch?** Describe it in `intent`, or add a launch
manifest or a driver script in `[workspace]`. [docs/protocol.md](docs/protocol.md) shows the
formats.

**Why does the trace span differ from the kernel time?** The trace covers the time between
the sites the agent placed; work before the first site, after the last one, and launch
overhead fall outside it. The analysis reports the gap.

## Contributing

Issues are welcome; pull requests are not accepted for now.
[CONTRIBUTING.md](CONTRIBUTING.md) says how to report a problem, and [AGENT.md](AGENT.md)
describes the layout, the tests and the rules a change follows.

## License

The MarsTrace client is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE): you may use, change and share it for any
noncommercial purpose. [NOTICE](NOTICE) holds the required notice. The license does not
allow commercial use, with one exception in
[LICENSE-ADDITIONAL-PERMISSION](LICENSE-ADDITIONAL-PERMISSION): a MarsCompute customer may use
the client for any purpose, commercial ones included, to access the MarsTrace service.
