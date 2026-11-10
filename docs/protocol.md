# From marstrace.toml to the MarsTrace API

The client turns a job directory into calls to the customer API of the MarsTrace service.
This document maps every key of `marstrace.toml` onto that API and lists the calls in the
order the client makes them.

## 1. Source of truth

Section 3 of the MarsTrace contract, `docs/reference/marstrace.md` of
mars-compute-ai/marscompute-maas, defines the API. Where this document and the contract
disagree, the contract wins.

The service publishes no JSON Schemas. The client checks every answer against its own
shapes in `marstrace/schema.py`: required fields, types, money strings, prefixed ids and
enumerations. A field the client does not read is accepted, so a service that adds fields
stays readable. The fake service in `tests/` refuses a request body with an unknown or a
missing field, as the service does.

## 2. The job directory

A job directory holds `marstrace.toml` and the files it names; every path stays inside the
directory. A job hands a workspace to an agent on the GPU host: cubins, and optionally a
source tree, a launch manifest, tensors and other files. The agent may read, build and run
what the workspace holds, within its budget, and returns one `marstrace.json`.

```toml
hardware = "b200"
intent = "find the longest bubble in the mainloop of the producer warps at M=N=K=8192"

[workspace]
cubins = ["k.cubin"]
source = "src"
manifest = "launch.json"
tensors = ["inputs.safetensors"]
files = ["run.py"]

[agent]
max_turns = 200
max_input_tokens = 20000000
max_output_tokens = 200000
max_wall_seconds = 2400

[limits]
max_cost_usd = 40.0
max_attempts = 2
```

## 3. Mapping

| `marstrace.toml` | Job request |
| --- | --- |
| `hardware` (required) | `hardware`, an `id` of `GET /v1/hardware` that is `available`; every cubin must target its `architecture` |
| `intent` (required, 1 to 32,000 characters, no NUL) | `prompt` |
| `[workspace] cubins` (1 to 8, each at most 64 MiB) | uploads of role `cubin` → `inputs.cubins` |
| `[workspace] source` (a directory) | archived as `source.tar.gz` (at most 256 MiB) → role `source_archive` → `inputs.source_archive` |
| `[workspace] manifest` (a `marscompute.kernel-launch` file, at most 1 MiB) | role `launch_manifest` → `inputs.launch_manifest` |
| `[workspace] tensors` (0 to 8 safetensors files, each at most 128 MiB) | role `tensor` → `inputs.tensors` |
| `[workspace] files` (0 to 8 files, each at most 16 MiB) | role `file` → `inputs.files` |
| `[agent] max_turns` (1 to 500), `max_input_tokens` (1 to 50,000,000), `max_output_tokens` (1 to 1,000,000), `max_wall_seconds` (60 to 2,580) | `agent`; an absent key takes `agent.defaults` of `GET /v1/hardware`, and none may exceed `agent.maximums` |
| `[limits] max_cost_usd` (required) | `limits.max_cost_micro_usd`, dollars converted exactly to a micro-USD string |
| `[limits] max_attempts` (1 to 3) | `limits.max_attempts`; absent, the service's default of 2 |

`[workspace]` and `[limits]` are required; any other key or table is refused. The agent's
model is not a request field: every job runs the model `GET /v1/hardware` names. A file is
named once in `[workspace]`, and all inputs together hold at most 256 MiB. A cubin is an
ELF cubin; PTX is refused. A launch manifest is JSON whose `schema_version` is
`marscompute.kernel-launch`. A tensors file is a safetensors file of the element types
`BOOL`, `U8`, `I8`, `F8_E4M3`, `F8_E5M2`, `U16`, `I16`, `F16`, `BF16`, `U32`, `I32`, `F32`,
`U64`, `I64` and `F64`.

The source archive is a gzip-compressed tar of the directory's files and subdirectories,
paths relative to it, with owners, times and modes normalized, so the same tree gives the
same archive. It leaves out every `.git` and `marstrace-results` entry.

`max_cost_usd` with more than 6 decimals is refused rather than rounded.

## 4. Calls

Every call carries `Authorization: Bearer <MARSTRACE_API_KEY>` (an `mt_live_...` key) to
`https://trace.marscompute.ai`, or to `MARSTRACE_API_URL` when it is set. Every answer is
the envelope `{"request_id", "data", "error"}`; the client reads `data` and raises `error`.

| Step | Call | Client |
| --- | --- | --- |
| hardware | `GET /v1/hardware` | `marstrace hardware` |
| credit | `GET /v1/credit` | `marstrace credit` |
| upload | `POST /v1/uploads` `{role, name, size_bytes, sha256, content_type}` → grant; `PUT` the bytes to `upload_url` with the grant's `headers`, or each 32 MiB range to its `parts[].upload_url` keeping each part's `ETag`; `POST /v1/uploads/{id}/complete` `{sha256, size_bytes, parts?}`, `parts` being `[{part_number, etag}]` | `marstrace estimate`, `marstrace submit` |
| quote | `POST /v1/quotes` with the job request | `marstrace estimate` |
| submit | `POST /v1/jobs` with the job request and `quote_id`, and an `Idempotency-Key` header | `marstrace submit` |
| state | `GET /v1/jobs/{id}` | `marstrace job` |
| wait | `GET /v1/jobs/{id}/wait?timeout_seconds=60`, repeated until the job ends | `marstrace job --wait`, `submit --wait` |
| list | `GET /v1/jobs?limit=&cursor=` → `items` and `next_cursor` | `marstrace jobs` |
| cancel | `POST /v1/jobs/{id}/cancel` with `{}` | `marstrace cancel` |
| result | `GET /v1/jobs/{id}/result` → `manifest`, `download_url` | `marstrace fetch` |

`estimate` and `submit` check the hardware and the size limits, then upload the workspace
and quote it. A grant's `x-amz-checksum-sha256` header must be the base64 of the file's
digest. A submission whose answer never arrived, or that the service answered with
`service_unavailable`, is sent again with the same key and body, which the service answers
with the job it started. `submit` records the request it sent under its key in
`marstrace-results/submissions/<key>.json`; submitting the directory again with the same
`--idempotency-key` resends that request without uploading again, so a lost answer is
recovered without a second job.

## 5. Results

`marstrace fetch` downloads `marstrace.json` from the presigned URL, checks it against the
digest and size of the result manifest's `file`, and writes it next to
`result-manifest.json`:

| File | Holds |
| --- | --- |
| `marstrace.json` | the trace report, read by `marstrace show` |
| `result-manifest.json` | the result manifest: `quality` (`usable`, `partial`, `unusable`), `reason_codes`, `file`, `device`, `clock_type` and `granularity` |

A job that ends without a usable trace still succeeds with `marstrace.json`, its
`quality` `unusable` and reason codes that say why. A download URL lasts 10 minutes, and
a result can be fetched for 30 days after the job ends.

## 6. Errors

A refusal answers the envelope with `error: {"code", "message", "retryable"}`. The client
prints the code, the message and the request id. The codes are those of section 3.2 of the
contract; the ones the client acts on:

| Code | Status | Client |
| --- | --- | --- |
| `insufficient_credit`, `spending_limit_exceeded`, `billing_suspended` | 402 | names the console page where credit is added, from `GET /v1/credit` |
| `idempotency_conflict` | 409 | the key was used with another request; submit with a fresh key |
| `price_changed` | 409 | the reservation exceeds `max_cost_micro_usd`; estimate again |
| `service_unavailable` | 503 | a submission is retried under the same key |

A failed job carries `error` with `code` and `message`.

## 7. Cost

A quote prices the request without reserving: the job fee, the GPU ceiling
(`(max_wall_seconds + 120) × 1000` milliseconds at the GPU rate), the agent's token rates
and `agent_budget_micro_usd`, which is `max_cost_micro_usd` less the fee and the GPU
ceiling. `admissible` is false when that budget is below $1; the client then refuses to
submit. Admission reserves at most `max_cost_micro_usd` against the workspace's credit,
and the job settles at what it used. `marstrace submit` prints the cap and the rates and
asks before it spends, unless `--yes`.

## 8. Local MCP server

`marstrace mcp` serves the commands as MCP tools over stdio (JSON-RPC 2.0, one message per
line), on local paths. The tools that call the service carry the names of the service's
remote MCP tools: `marstrace_hardware`, `marstrace_credit`, `marstrace_estimate`,
`marstrace_submit`, `marstrace_jobs`, `marstrace_job`, `marstrace_wait`,
`marstrace_cancel` and `marstrace_result`. `marstrace_estimate` and `marstrace_submit`
take a job directory and upload it themselves, so the remote `marstrace_upload` and
`marstrace_upload_complete` have no local counterpart. The local tools are
`marstrace_cubin_find`, `marstrace_cubin_show`, `marstrace_fetch` and `marstrace_show`.
`marstrace_submit` takes `accept_max_cost_usd` and refuses a quote whose cap is above it.
`.mcp.json` registers the server for Claude Code.
