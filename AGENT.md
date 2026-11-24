# Agent Instructions

This file is for a code agent that changes the MarsTrace client, and for one that uses it to
trace a kernel. An agent that only uses the client should read
[Using the client](#using-the-client) and load the skills in `marstrace/skills/`.

## Repository layout

- `marstrace/`: the `marstrace` command.
  - `job.py`: `marstrace.toml` and its translation into a job request.
  - `binary.py`: cubins and compiler caches.
  - `tensors.py`: safetensors files.
  - `client.py`: the HTTP client of the MarsTrace API.
  - `schema.py`: the client's check of every answer.
  - `workflow.py`: the operations the commands and the MCP server share.
  - `analysis.py`: the delivered analysis, read for a person.
  - `mcp.py`: the stdio MCP server.
  - `cli.py`: the commands.
- `marstrace/viewer/`: `marstrace show`. It is generated; see the constraints below.
- `marstrace/skills/`: the `marstrace-remote` and `marstrace-read` skills. The wheel,
  `.claude-plugin/` and `.mcp.json` package them with the MCP server for Claude Code.
- `docs/protocol.md`: how `marstrace.toml` maps onto the MarsTrace API, and what the results hold.
- `examples/matmul/`: a job built from `src/matmul.cu` with the command in its
  `marstrace.toml`; the tests copy it into every job they build.
- `tests/`: unit tests, the fake service (`fake_service.py`) and fixtures.

## Running the tests

```bash
pip install -e .
python -m unittest discover -s tests
ruff check .
```

Tests never call the real service. They run against `tests/fake_service.py`, which refuses a
request body with an unknown or missing field, and the fixtures in `tests/fixtures/`. Its
rate card holds fictional numbers; keep them fictional.

## Conventions

A change follows the Python, README and git conventions of
[humanize](https://github.com/zobinHuang/humanize). Commit messages are one line in
Conventional Commits form, with no agent attribution.

## Constraints

- **The client carries no tracing implementation.** It holds no instrumentation or
  scheduling logic, and no text that says how tracing works on the service. Execution
  stays on the service.
- **Customer-visible text names the product MarsTrace only.** Messages, help, skills,
  docs and the delivered files never name the tracing engine, a supplier, an internal tool
  or an earlier name of the product.
- **Nothing private is committed.** No API keys or tokens (`mt_live_...`, model keys, cloud
  credentials), no `.env` files, no internal host names, addresses or repository names, no
  internal costs or real prices, and no customer data in fixtures. Tests and examples use
  fictional rates and our own sample kernels.
- **Code reaches the service only through `[workspace]`.** Its `source` and `files` are what
  the user shares with the agent on the GPU machine, on purpose; add no other path that
  carries code to execute.
- **The viewer is generated.** Do not edit `marstrace/viewer/` by hand: the maintainers
  regenerate it from their own viewer, and a hand edit is lost. A fix to the viewer goes to
  the maintainers. A view only the client has, such as `show --analysis`, lives in the
  client's modules (`DeliveredAnalysis.extend_show` in `analysis.py`) and is attached to
  `show` after the viewer registers it.
- **The service's API is the source of truth.** A change to the API changes `client.py`,
  `schema.py`, `job.py`, the fake service and `docs/protocol.md` together.

## Compatibility

- **Old clients keep working against a newer service.** The service only adds fields and
  values; a client accepts and ignores fields it does not know, and `schema.py` checks only
  what the client reads. When an answer gains a field, make it optional in `schema.py`.
- **A new client sends what an older service accepts.** A new request field is optional,
  or sent only when the manifest sets it, until the service has accepted it in production.
- **`marstrace.toml` refuses unknown keys**, so a misspelled key fails locally. A new key
  goes into `JobManifest.SET_KEY`, its check, `docs/protocol.md` and a test.

## Releasing

1. Bump the version in `pyproject.toml`, `SERVER_VERSION` in `marstrace/mcp.py` and
   `.claude-plugin/plugin.json`, in a `chore(release): bump version to X.Y.Z` commit.
2. Merge the pull request after its tests pass.
3. Tag the merge commit `vX.Y.Z` with an annotated tag and push the tag.
4. Install the tag into a fresh environment
   (`pip install 'git+https://github.com/mars-compute-ai/MarsTrace@vX.Y.Z'`) and check
   `pip show mars-trace` and `marstrace --help`.
5. Update the version the README's quick start installs.

## Using the client

- **Through MCP:** `marstrace_hardware`, `marstrace_estimate`, `marstrace_submit`,
  `marstrace_job`, `marstrace_fetch` and `marstrace_show` take local paths. Estimate first.
  Show the user the rates and the cost at the bounds, and submit only with their yes
  (`accept_cost_at_bounds_usd`). Reuse the same `idempotency_key` when retrying a submission.
- **Gate the result first.** Read `quality` and `reason_codes` in `result-manifest.json`
  before any number. `unusable` has no timeline; `partial` limits what each reason code
  names.
- **Read the analysis by its rules.** `marstrace_fetch` answers it, and `marstrace show
  <file> --analysis` prints it:
  - quote the findings' numbers as they read;
  - name where each finding sits from its Where line (SASS index, opcode, source line);
  - a hypothesis, or an item under what the trace cannot tell, is not a cause until its
    measurement has been made, so say which measurement would confirm it;
  - take the blocks resident per SM from the analysis, and never derive waves from blocks
    divided by SMs;
  - a traced span shorter than the kernel time means work outside the traced sites or
    launch overhead, which the UNSEEN finding quantifies.
- **Read the trace with `marstrace show`**, not a hand-written JSON parser. The skill
  `marstrace-read` says which view answers which question.
