# xtrace-harness

xtrace-harness traces the timeline inside a CUDA kernel on MarsCompute GPUs, and reads the
trace in your terminal. You send the kernel's cubin with a description of its launch, or
hand an agent on the GPU host a workspace of cubins and source to trace on its own budget;
MarsCompute returns a report of per-region durations and pipeline bubbles. It is for kernel
developers and the coding agents that work with them; no local GPU is needed.

## Usage

Create an API key in the [MarsCompute console](https://console.marscompute.ai) and add
credit, then:

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
export MARSCOMPUTE_API_KEY=mc_live_v1_...

xtrace hardware                                        # hardware, rates, architectures
xtrace cubin find my_triton_kernel --copy-to my-job    # or nvcc -cubin; see the skill
xtrace inspect examples/matmul/matmul.cubin --kernel square_matrix_mul   # quotes a fee, asks
xtrace sass examples/matmul/matmul.cubin --kernel square_matrix_mul --grep STG
xtrace estimate examples/matmul                        # the cost cap and the rate
xtrace submit examples/matmul --wait
xtrace show examples/matmul/xtrace-results/<job>/xtrace.json --stats --bubbles
```

For Claude Code, install the plugin, which adds the `xtrace mcp` server and the
`xtrace-remote` and `xtrace-read` skills:

```bash
claude plugin marketplace add mars-compute-ai/xtrace-harness
claude plugin install xtrace@marscompute
```

Other agents can load the skills from [skills/](skills/) and run `xtrace mcp` as a stdio
MCP server.

[docs/protocol.md](docs/protocol.md) maps `xtrace.toml`, agent mode included, onto the
platform API.
