# From xtrace.toml to the Xtrace API

The harness turns a job directory into calls to the Xtrace API of the MarsCompute Control
Plane. This document maps every key of `xtrace.toml` onto that API and lists the calls in
the order the harness makes them.

## 1. Source of truth

The platform's contracts define the API: the zod schemas in `packages/contracts/src/`
(`xtrace.ts`, `xtrace-launch-manifest.ts`) of mars-compute-ai/marscompute-maas, exported
as JSON Schema to `packages/contracts/json-schema/xtrace/*.schema.json`. Agent mode is
specified in `docs/reference/xtrace-agent.md` of the same repository. Where this document
and those sources disagree, the sources win.

The harness vendors the exported schemas into `xtrace_harness/schemas/`:

```bash
python scripts/sync_schemas.py <maas checkout>           # copy them
python scripts/sync_schemas.py <maas checkout> --check   # exit 1 when the copy is stale
```

The tests validate every request body the client builds against the vendored schema of
its route (`upload-begin`, `upload-complete`, `estimate-input`, `submit-input`), every
launch manifest uploaded against `launch-manifest`, and the fake Control Plane's
capabilities, credit, quotes, jobs and result manifests against theirs. A schema that is
not vendored is skipped.

## 2. The job directory

A job directory holds `xtrace.toml` and the files it names; every path stays inside the
directory. A job hands a workspace to an agent on the GPU host: cubins, and optionally a
source tree, a launch manifest, tensors and other files. The agent may read, build and
run what the workspace holds, within its budget, and returns one `xtrace.json`.

```toml
hardware = "h200-cuda12.9"
intent = "find the longest bubble in the mainloop of the producer warps at M=N=K=8192"

[workspace]
cubins = ["k.cubin"]
source = "src"
manifest = "launch.json"
tensors = ["inputs.safetensors"]
files = ["run.py"]

[agent]
model = "claude-opus-5-5"
max_turns = 200
max_input_tokens = 20000000
max_output_tokens = 200000
max_wall_seconds = 2400

[limits]
max_cost_usd = 40.0
max_gpu_seconds = 3600
```

## 3. Mapping

| `xtrace.toml` | API |
| --- | --- |
| `hardware = "<environment_id>"` (required) | `compute` of the capabilities entry with that `environment_id`; every cubin must target its `architecture` |
| `intent` (required, at most 32,000 characters) | `prompt` |
| `[workspace] cubins` (1 to 8) | artifacts kind `cubin` → `bundle.cubin_artifact_ids` |
| `[workspace] source` (a directory) | archived as `source.tar.gz` → kind `source` → `bundle.source_archive_artifact_id` |
| `[workspace] manifest` (a `marscompute.kernel-launch` file) | kind `launch_manifest` → `bundle.launch_manifest_artifact_id` |
| `[workspace] tensors` (0 to 8 safetensors files) | kind `safetensors` → `bundle.tensor_artifact_ids` |
| `[workspace] files` (0 to 8 files) | kind `workspace_file` → `bundle.file_artifact_ids` |
| `[agent] model` | `analysis.model`, one of the capabilities' `agent_models` |
| `[agent] max_turns` (1 to 500), `max_input_tokens` (1 to 50,000,000), `max_output_tokens` (1 to 1,000,000), `max_wall_seconds` (60 to 2580) | `analysis` `{mode: "agent", model, max_turns, max_input_tokens_total, max_output_tokens_total, max_wall_seconds}` |
| `[limits] max_cost_usd, max_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes` | `limits` `{max_total_cost_micro_usd, max_active_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes}`; an absent key takes the capabilities' `limits` |

`bundle.kind` is `agent_workspace`, and the request's `schema_version` is
`marscompute.xtrace.submit.v1`. `[workspace]` and `[agent]` are required; any other key or
table is refused. A file is named once in `[workspace]`. A cubin is an ELF cubin; PTX is
refused. A launch manifest is JSON whose `schema_version` is `marscompute.kernel-launch`.
A tensors file is a safetensors file of the element types `BOOL`, `U8`, `I8`, `F8_E4M3`,
`F8_E5M2`, `U16`, `I16`, `F16`, `BF16`, `U32`, `I32`, `F32`, `U64`, `I64` and `F64`.

The source archive is a gzip-compressed tar of the directory's files and subdirectories,
paths relative to it, with owners, times and modes normalized, so the same tree gives the
same archive. It leaves out every `.git`, `.xtrace-cache` and `xtrace-results` entry.

`max_cost_usd` is rounded down to whole micro-USD and must be positive.
`max_gpu_seconds` may not exceed the effective `max_wall_seconds`.

## 4. Calls

Every call carries `Authorization: Bearer <MARSCOMPUTE_API_KEY>` (an `mc_live_v1_...` key)
to `https://control.marscompute.ai`, or to `MARSCOMPUTE_API_URL` when it is set.

| Step | Call | Harness |
| --- | --- | --- |
| hardware | `GET /v1/xtrace/capabilities` | `xtrace hardware` |
| credit | `GET /v1/xtrace/credit` | `xtrace credit` |
| upload | `POST /v1/xtrace/artifacts/uploads` `{kind, name, size_bytes, sha256, content_type}` → grant; `PUT` the bytes to `upload_url`, or each range to its `parts[].upload_url` keeping each part's `ETag`; `POST /v1/xtrace/artifacts/{id}/complete` `{sha256, size_bytes, parts?}` | `xtrace estimate`, `xtrace submit` |
| quote | `POST /v1/xtrace/quotes` with the estimate input | `xtrace estimate` |
| submit | `POST /v1/xtrace/jobs` with the estimate input, `quote_id` and `idempotency_key`, and the same key in `Idempotency-Key` | `xtrace submit` |
| state | `GET /v1/xtrace/jobs/{id}`, polled every `poll_after_milliseconds` | `xtrace job`, `--wait` |
| list | `GET /v1/xtrace/jobs?limit=&cursor=` | `xtrace jobs` |
| cancel | `POST /v1/xtrace/jobs/{id}/cancel` with `{}` | `xtrace cancel` |
| result | `GET /v1/xtrace/jobs/{id}/result` → `manifest` and `downloads` | `xtrace fetch` |

`estimate` and `submit` upload the workspace before they quote; a file larger than the
capabilities' `max_upload_bytes` is refused before it is sent. A submission whose answer
never arrived is sent again with the same idempotency key, which never starts a second
job.

## 5. Results

`xtrace fetch` checks every file against its download grant and the result manifest,
decompresses gzip-encoded files, and writes them by name next to `result-manifest.json`:

| File | Holds |
| --- | --- |
| `xtrace.json` | the trace report, read by `xtrace show` |
| `result-manifest.json` | the result manifest: status, `quality` (`usable`, `partial`, `unusable`), `reason_codes`, files and environment |

A session that ends without a trace still returns `xtrace.json`, with `quality:
"unusable"` and reason codes that say why.

## 6. Errors

A refusal answers `{"error": {"code", "message", "correlation_id"}}`. The harness prints
the code, the message and the correlation id.

| Code | Status | Meaning |
| --- | --- | --- |
| `insufficient_credit` | 402 | the credit cannot hold the quote; the error carries `top_up_url`, the console page where credit is added |
| `invalid_payload` | 400 | a body does not match its schema |

A failed job carries `error` with `code`, `message`, `classification` and `retryable`.

## 7. Cost

A quote freezes `limits.max_total_cost_micro_usd`, the most the job can cost, and the
rates of `rate_snapshot`: the GPU rate (`gpu_micro_usd_per_hour`), the agent model's token
rates per million input, cached input and output tokens (`model_rates`), and the job fee
and cache write rates (`price_book`). The cap must cover the job fee, the tokens at their
caps and GPU time at `max_active_gpu_seconds`; the GPU is held while the agent works, so
that time is billed. Admission holds the cap against the tenant's credit, and the job
settles at what it used. `xtrace submit` prints the cap and the rates and asks before it
spends, unless `--yes`.

## 8. Local MCP server

`xtrace mcp` serves the commands as MCP tools over stdio (JSON-RPC 2.0, one message per
line), on local paths: `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_estimate`, `xtrace_submit`, `xtrace_jobs`, `xtrace_job`,
`xtrace_wait`, `xtrace_cancel`, `xtrace_fetch` and `xtrace_show`. `xtrace_submit` takes
`accept_max_cost_usd` and refuses a quote whose cap is above it. `.mcp.json` registers the
server for Claude Code.
