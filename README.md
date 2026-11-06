# xtrace-harness

xtrace-harness traces the timeline inside a CUDA kernel on MarsCompute GPUs, and reads the
trace in your terminal. You send the kernel's cubin with a description of its launch;
MarsCompute runs it under Xtrace on a GPU of the class you choose and returns a report of
per-region durations and pipeline bubbles. It is for kernel developers and the coding
agents that work with them; no local GPU is needed.

## Usage

Create an API key in the [MarsCompute console](https://console.marscompute.ai) and add
credit, then:

```bash
pip install git+https://github.com/mars-compute-ai/xtrace-harness
export MARSCOMPUTE_API_KEY=mc_live_...

xtrace hardware                                   # GPU classes and prices
xtrace cubin find my_triton_kernel --copy-to my-job  # or nvcc -cubin; see the skill
xtrace submit examples/matmul --kind inspect --wait
xtrace sass examples/matmul/xtrace-results/<job>/inspect.json --line matmul.cu:8
xtrace submit examples/matmul --wait
xtrace show examples/matmul/xtrace-results/<job>/trace.json --stats --bubbles
```

For Claude Code, install the plugin, which adds the MarsCompute MCP server and the
`xtrace-remote` and `xtrace-read` skills:

```bash
claude plugin marketplace add mars-compute-ai/xtrace-harness
claude plugin install xtrace@marscompute
```

Other agents can load the skills from [skills/](skills/) and point their MCP client at
`https://api.marscompute.ai/mcp` with the API key as a bearer token.

[docs/protocol.md](docs/protocol.md) describes the job manifest, the interfaces and the
result files.
