# From marstrace.toml to the MarsTrace API

The client turns a job directory into calls to the customer API of the MarsTrace service.
This document maps every key of `marstrace.toml` onto that API and lists the calls in the
order the client makes them.

## 1. Source of truth

Section 3 of the MarsTrace contract, `docs/reference/marstrace.md` of
mars-compute-ai/marscompute-maas, defines the API, and the service in
`services/marstrace` of the same repository implements it. Where this document and the
contract disagree, the contract wins; where the contract and the service disagree, the
client follows the service.

The service publishes no JSON Schemas. The client checks every answer against its own
shapes in `marstrace/schema.py`: required fields, types, money strings, prefixed ids, times
and enumerations. A field the client does not read is accepted, so a service that adds
fields stays readable. The fake service in `tests/` mirrors the service's routes, request
schemas, answers, envelopes and error codes: it refuses a request body with an unknown
field, a missing required field or a value out of range, as the service does.

## 2. The job directory

A job directory holds `marstrace.toml` and the files it names; every path stays inside the
directory. A job hands a workspace to an agent on the GPU host: cubins, and optionally a
source tree, a launch manifest, tensors and other files. The agent may read, build and run
what the workspace holds, within the caps of `[agent]`, and returns one `marstrace.json`.

```toml
hardware = "b200"
cuda_version = "13.0"
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
max_attempts = 2
```

## 3. Mapping

| `marstrace.toml` | Job request |
| --- | --- |
| `hardware` (required) | `hardware`, an `id` of `GET /v1/hardware` that is `available`; every cubin must target its `architecture` |
| `cuda_version` (optional, `M.m` such as `13.0`) | `cuda_version`, one of the hardware's `cuda_versions` in `GET /v1/hardware`; absent, the request has no `cuda_version` and the job runs the hardware's `default_cuda_version`. The agent's CUDA toolkit is that version, and the result manifest's `device.cuda_toolkit_version` names it |
| `intent` (required, 1 to 32,000 characters, no NUL) | `prompt` |
| `[workspace] cubins` (1 to 8, each at most 64 MiB) | uploads of role `cubin` → `inputs.cubins` |
| `[workspace] source` (a directory) | archived as `source.tar.gz` (at most 256 MiB) → role `source_archive` → `inputs.source_archive` |
| `[workspace] manifest` (a `marscompute.kernel-launch` file, at most 1 MiB) | role `launch_manifest` → `inputs.launch_manifest` |
| `[workspace] tensors` (0 to 8 safetensors files, each at most 128 MiB) | role `tensor` → `inputs.tensors` |
| `[workspace] files` (0 to 8 files, each at most 16 MiB) | role `file` → `inputs.files` |
| `[agent] max_turns` (1 to 500), `max_input_tokens` (1 to 50,000,000), `max_output_tokens` (1 to 1,000,000), `max_wall_seconds` (60 to 2,580) | `agent`; an absent key takes `agent.defaults` of `GET /v1/hardware`, and none may exceed `agent.maximums` |
| `[limits] max_attempts` (1 to 3) | `limits.max_attempts`; absent, the request has no `limits` and the service takes 2 |

`[workspace]` is required, `cuda_version`, `[agent]` and `[limits]` are optional, and any
other key or table is refused. `marstrace estimate` and `submit` take `--cuda-version` to
override `cuda_version`, and the MCP tools take `cuda_version`; the client checks the
version against the hardware before it uploads anything. The request carries no amount of
money: a job is charged the agent tokens it actually used (section 7). The agent's
model is not a request field: every job runs the model `GET /v1/hardware` names. A file is
named once in `[workspace]`, and all inputs together hold at most 256 MiB. A cubin is an
ELF cubin; PTX is refused. A launch manifest is JSON whose `schema_version` is
`marscompute.kernel-launch`. A tensors file is a safetensors file of the element types
`BOOL`, `U8`, `I8`, `F8_E4M3`, `F8_E5M2`, `U16`, `I16`, `F16`, `BF16`, `U32`, `I32`, `F32`,
`U64`, `I64` and `F64`.

The source archive is a gzip-compressed tar of the directory's files and subdirectories,
paths relative to it, with owners, times and modes normalized, so the same tree gives the
same archive. It leaves out every `.git` and `marstrace-results` entry.

## 4. Calls

Every call carries `Authorization: Bearer <MARSTRACE_API_KEY>` (an `mt_live_...` key) to
the service URL `https://console.marscompute.ai/trace`, or to `MARSTRACE_API_URL` when it is
set. Each path below is appended to that URL, so `GET /v1/hardware` goes to
`https://console.marscompute.ai/trace/v1/hardware`. Every answer is the envelope
`{"request_id", "data", "error"}`; the client reads `data` and raises `error`.

| Step | Call | Client |
| --- | --- | --- |
| hardware | `GET /v1/hardware` → the service's token `rates`, the same on every hardware, and per hardware its `architecture`, `cuda_versions` and `default_cuda_version` | `marstrace hardware` |
| credit | `GET /v1/credit` | `marstrace credit` |
| upload | `POST /v1/uploads` `{role, name, size_bytes, sha256, content_type}` → grant; `PUT` the bytes to `upload_url` with the grant's `headers`, or each 32 MiB range to its `parts[].upload_url` keeping each part's `ETag`; `POST /v1/uploads/{id}/complete` `{sha256, size_bytes, parts?}`, `parts` being `[{part_number, etag}]` | `marstrace estimate`, `marstrace submit` |
| estimate | `POST /v1/quotes` with the job request; the service stores and holds nothing | `marstrace estimate` |
| submit | `POST /v1/jobs` with the job request and an `Idempotency-Key` header | `marstrace submit` |
| state | `GET /v1/jobs/{id}` | `marstrace job` |
| wait | `GET /v1/jobs/{id}/wait?timeout_seconds=60`, repeated until the job ends | `marstrace job --wait`, `submit --wait` |
| list | `GET /v1/jobs?limit=&cursor=` → `items` and `next_cursor` | `marstrace jobs` |
| cancel | `POST /v1/jobs/{id}/cancel` with `{}` | `marstrace cancel` |
| result | `GET /v1/jobs/{id}/result` → `manifest`, `download_url` | `marstrace fetch` |

`estimate` and `submit` check the hardware, the CUDA version and the size limits, then
upload the workspace and price it. The estimate and the job carry the CUDA version the job
runs with, the default one when the request names none. A grant's `x-amz-checksum-sha256` header must be the base64 of the file's
digest. The service keys a submission by the account, the `Idempotency-Key` and the
SHA-256 of the body as sent, with keys sorted, for 24 hours: the same key and body answer
the job it started, and another body is `idempotency_conflict`. A submission whose answer
never arrived, or that the service answered with `service_unavailable`, is sent again with
the same key and body. `submit` records the request it sent under its key in
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

A refusal answers the envelope with `error: {"code", "message", "retryable", "details"?}`;
a refusal for credit carries `details.add_credit_url`, and `rate_limited` and
`too_many_jobs` set the `retry-after` header. The client prints the code, the message, the
credit page, the wait and the request id. The codes are those of section 3.2 of the
contract:

| Code | Status | Client |
| --- | --- | --- |
| `invalid_request` | 400 | the message names the field; a CUDA version the hardware does not offer names the ones it does |
| `api_key_missing` | none | raised by the client, before any call, when `MARSTRACE_API_KEY` holds no `mt_live_` key; names the console steps that give one |
| `authentication_required` | 401 | set `MARSTRACE_API_KEY` to a live key |
| `insufficient_credit`, `spending_limit_exceeded`, `billing_suspended` | 402 | the available balance is below the minimum a job needs, or billing is not active; names the console page where credit is added |
| `forbidden` | 403 | the account or workspace is not active for MarsTrace |
| `not_found` | 404 | no such job or upload for this account |
| `idempotency_conflict` | 409 | the key was used with another request; submit with a fresh key |
| `upload_not_ready`, `job_not_cancellable`, `result_unavailable` | 409 | an upload is not verified or belongs to another job; the job has ended; the job has no result |
| `limit_exceeded` | 413 | a size or count limit |
| `hardware_unavailable` | 422 | the hardware is unknown, or MarsTrace has no published price |
| `rate_limited`, `too_many_jobs` | 429 | retryable after `retry-after` seconds |
| `service_unavailable` | 503 | retryable; a submission is retried under the same key |

A failed job carries `error` with `code` and `message`, and each failed attempt its
`failure` class (`supplier`, `platform`, `workload` or `cancel`) and `error`.

## 7. Cost

A job is charged the agent tokens it actually used, whatever its outcome: the input, output,
cache-read and cache-write tokens of every attempt, retried and failed attempts included,
at the service's token rates, the same on every hardware. A job whose agent never ran is charged
nothing. The usage is charged in full even when it passes the available balance.

An estimate (`POST /v1/quotes`) answers:

| Field | Holds |
| --- | --- |
| `cuda_version` | the CUDA version the job runs with |
| `rates` | the agent's token rates per million: input, cache read, 5-minute and 1-hour cache write, output |
| `bounds_per_attempt` | the agent's `agent_input_tokens` and `agent_output_tokens` caps |
| `cost_at_bounds_micro_usd` | `agent_tokens_per_attempt` and `total`, `max_attempts` attempts at their token bounds: the most the job can cost |
| `minimum_micro_usd` | the balance a job needs to start, set by MarsCompute administrators |
| `available_micro_usd` | the workspace's available balance; negative while it owes for a job that ran past it, null without a spending limit |
| `admissible`, `message` | whether a job would start now, and why not |

Nothing is held against the cost at the bounds. Admission holds the minimum until the job
settles (`cost.held_micro_usd` of the job), so a workspace runs as many jobs at once as its
balance holds minimums. A job's `progress.usage_micro_usd` is its usage so far; once it
settles, `cost.charged_micro_usd` and `cost.lines`, `{dimension, quantity,
amount_micro_usd}` per token dimension of the rate card, hold the charge, which may exceed the hold.
`marstrace submit` prints the CUDA version, the token rates, the cost at the bounds and the balance, refuses a job
that is not admissible, and asks before it submits, unless `--yes`.

## 8. Local MCP server

`marstrace mcp` serves the commands as MCP tools over stdio (JSON-RPC 2.0, one message per
line), on local paths. The tools that call the service carry the names of the service's
remote MCP tools: `marstrace_hardware`, `marstrace_credit`, `marstrace_estimate`,
`marstrace_submit`, `marstrace_jobs`, `marstrace_job`, `marstrace_wait`,
`marstrace_cancel` and `marstrace_result`. `marstrace_estimate` and `marstrace_submit`
take a job directory and upload it themselves, so the remote `marstrace_upload` and
`marstrace_upload_complete` have no local counterpart. The local tools are
`marstrace_guide`, `marstrace_cubin_find`, `marstrace_cubin_show`, `marstrace_fetch` and
`marstrace_show`; `marstrace_guide` returns the skills `marstrace-remote` (topic `remote`) and
`marstrace-read` (topic `read`) that the package ships, as `marstrace guide` prints them.
The server starts without `MARSTRACE_API_KEY`: the local tools answer, and a tool that calls
the service answers `api_key_missing` with the steps that give the user a key.
`marstrace_show` with `--json` answers the JSON alone.
`marstrace_submit` takes `accept_cost_at_bounds_usd`, the cost at the bounds the user
agreed to, and refuses an estimate whose cost at the bounds is above it, since prices are
read again at submission.
`.mcp.json` registers the server for Claude Code.
