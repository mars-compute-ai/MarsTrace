# mars-trace

mars-trace is the client of MarsTrace, which traces the timeline inside a CUDA kernel on an
H200, B200 or B300 and reads the trace in your terminal. You hand an agent on the GPU host
a workspace of the kernel's cubins, and optionally its source and data; it traces the
kernel within the caps you set and returns `marstrace.json`, a report of per-region
durations and pipeline bubbles. It is for kernel developers and the coding agents that
work with them; no local GPU is needed.

## Usage

Create a MarsTrace API key in the [MarsCompute console](https://console.marscompute.ai)
and add credit; a job starts only while the workspace's balance reaches the minimum that
`marstrace credit` shows, and it is charged the GPU time and agent tokens it used. Then:

```bash
pip install git+https://github.com/mars-compute-ai/mars-trace
export MARSTRACE_API_KEY=mt_live_...

marstrace hardware                                        # hardware, rates, agent model
marstrace cubin find my_triton_kernel --copy-to my-job    # or nvcc -cubin; see the skill
marstrace estimate examples/matmul                        # rates, the most the job can cost, the balance
marstrace submit examples/matmul --wait
marstrace show examples/matmul/marstrace-results/<job>/marstrace.json --stats --bubbles
```

For Claude Code, install the plugin, which adds the `marstrace mcp` server and the
`marstrace-remote` and `marstrace-read` skills:

```bash
claude plugin marketplace add mars-compute-ai/mars-trace
claude plugin install marstrace@marscompute
```

Other agents can load the skills from [skills/](skills/) and run `marstrace mcp` as a
stdio MCP server. [docs/protocol.md](docs/protocol.md) maps `marstrace.toml` onto the
MarsTrace API.
