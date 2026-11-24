# MarsTrace

MarsTrace times the inside of a CUDA kernel on a cloud GPU. You send the kernel's cubin, and
an agent on a GPU machine instruments it, runs it at the launch you describe and returns
`marstrace.json`: per-region durations for every thread or warp, the balance of the grid, and
an analysis of where the time goes. This repository is the client, the `marstrace` command
and its MCP server. It is for kernel developers and their coding agents, and it needs no
local GPU.

[中文版](README.zh.md)

## Quick start

1. Register at the [MarsCompute console](https://console.marscompute.ai/register), add
   credit, and create an API key under MarsTrace › Keys.
2. Install the client and trace the bundled example:

```bash
pip install 'git+https://github.com/mars-compute-ai/MarsTrace@v0.2.0'
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
It does not change your kernel either.

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

- Mode `fast`: a trace and its analysis. The modes `thinking-light` and `thinking-heavy`,
  which will optimize a kernel, are not offered yet.
- GPUs: H200 and B200.
- CUDA 13.0.
- Inputs: up to 8 cubins (ELF cubins, not PTX), plus optional source, a launch manifest,
  safetensors inputs and other files, 256 MiB in all.

## Billing

A job is charged the agent's tokens over all its attempts, at the rates `marstrace hardware`
and `marstrace estimate` show. GPU time is not charged separately. A job whose trace is
`unusable`, or that fails without delivering a result, is not charged. A cancelled job pays
for the tokens used up to the stop. A job starts only while your balance reaches the
minimum that `marstrace credit` shows, and `marstrace estimate` shows the most it can cost.

## Privacy and security

- Each attempt runs on its own GPU machine, which is deleted when the attempt ends. Nothing
  is shared between jobs or customers.
- The machine accepts no inbound connection, and the agent's container reaches nothing but
  the MarsTrace service.
- Your uploaded files are deleted 24 hours after the job ends. Results can be downloaded
  for 30 days.
- MarsCompute keeps the agent's model requests and replies for each job, which may quote
  your code. Only MarsCompute administrators can read them, and they are deleted on your
  request.

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

Issues and pull requests are welcome. [AGENT.md](AGENT.md) describes the layout, the tests
and the rules a change follows; run `python -m unittest discover -s tests` before you open a
pull request.

## License

The license will be added before this repository is published.
