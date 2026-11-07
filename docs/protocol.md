# From xtrace.toml to the Xtrace API

The harness turns a job directory into calls to the Xtrace API of the MarsCompute Control
Plane. This document maps every key of `xtrace.toml` onto that API and lists the calls in
the order the harness makes them.

## 1. Source of truth

The platform's contracts define the API: the zod schemas in
`packages/contracts/src/xtrace.ts` of mars-compute-ai/marscompute-maas, exported as JSON
Schema to `packages/contracts/json-schema/xtrace/*.schema.json`. Where this document and
those schemas disagree, the schemas win.

The harness vendors the exported schemas into `xtrace_harness/schemas/`:

```bash
python scripts/sync_schemas.py <maas checkout>           # copy them
python scripts/sync_schemas.py <maas checkout> --check   # exit 1 when the copy is stale
```

The tests validate every request body the client builds, and every launch manifest it
uploads, against the vendored schema of the same name (`upload-begin`, `upload-complete`,
`inspection-request`, `estimate-input`, `submit-input`, `launch-manifest`). A schema that
is not vendored yet is skipped.

## 2. The job directory

A job is data: a cubin, a description of its launch, the buffers' contents and what to
measure. The platform runs no program of the user's.

| File | Holds | Uploaded as |
| --- | --- | --- |
| `xtrace.toml` | the manifest | translated, see section 3 |
| the cubin | the kernel; PTX is not accepted | artifact `cubin` |
| the sources | optional; the viewer shows them beside the listing | not uploaded |
| the tensors file | optional safetensors with the contents of `init = "tensor"` buffers | artifact `safetensors` |
| the reference file | optional safetensors with the expected contents of the `[validation]` output | artifact `reference` |

Every path stays inside the directory. `.xtrace-cache/` beside a cubin holds its cached
inspections, keyed by the cubin's sha256 and the kernel symbol.

## 3. Mapping

| `xtrace.toml` | API |
| --- | --- |
| `hardware = "<environment_id>"` | `compute` = the `compute` of the capabilities entry with that `environment_id`; its `architecture` becomes `target.architecture` |
| `intent` (required) | `prompt` |
| `[kernel] binary` | artifact kind `cubin` → `bundle.cubin_artifact_id` |
| `[kernel] name` | `target.kernel_symbol` |
| `[kernel] sources` | not uploaded |
| `tensors` | artifact kind `safetensors` → `bundle.tensors_artifact_id` |
| `reference` + `[validation]` (`output`, `atol`, `rtol`) | artifact kind `reference` → `bundle.reference_artifact_id`; manifest `validation` `{output_key, atol, rtol}` |
| `[[buffers]] name, dtype, shape, init, seed` | manifest `buffers[]` `{id, dtype, shape, init, seed \| tensor_key}` |
| `[[buffers]] output = true` | `buffer` arguments that name it get `access = "read_write"`, others `"read"` |
| `[launch] grid, block, cluster, shared_bytes` | `target.grid`, `target.block`, `target.cluster` (then `launch_kind = "cluster"`, else `"ordinary"`), `target.dynamic_shared_memory_bytes` |
| `[launch] args` | `arguments[]` with `index` = position |
| `[[trace.sites]] index` | `sites[].instruction_index` |
| `[[trace.sites]] line` | the first instruction of that line in the inspection's listing that is in `eligible_sites` |
| `value_register`, `value_width_bits` | `probe = "value"`, `value_register`, `value_width_bits`; `capture_values = true` |
| `value_is_smid` | `probe = "smid"`; `capture_values = true` |
| `warp_level` (alike on every site) | `granularity` `"warp"`, else `"thread"` |
| `per_thread_records` (default 32) | `max_records` = records × rows; rows are the launch's threads, or its warps |
| `[trace] clock`, `schedule`, `[trace.scope_roles]` | `clock`, `schedule`, `scope_roles` |
| `[[trace.regions]] enter, leave, label` | `regions[]` `{region_id = label, start_site_id = "s<enter>", end_site_id = "s<leave>"}` |
| `[trace] stages` with `[trace.range_profile]`, `[trace.pc_sampling]` | `stages`, `range_profile`, `pc_sampling` |
| `[planner] model, max_calls, max_input_tokens, max_output_tokens` (instead of `[trace]`) | `analysis = {mode: "platform_planned", model, max_calls, max_input_tokens_total, max_output_tokens_total, capture_values: false}` |
| `[limits] max_cost_usd, max_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes` | `limits` `{max_total_cost_micro_usd, max_active_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes}`; an absent key takes the capabilities' `limits` |

Site `n` is the n-th `[[trace.sites]]` entry and has `site_id = "s<n>"`. A region label is
1 to 200 letters, digits, `_`, `.`, `:` or `-`; an unlabelled region is `s<enter>-s<leave>`.
`[trace]` with sites becomes `analysis = {mode: "explicit_plan", trace_plan}` with
`tier = "sass"`.

### 3.1 Element types

| `dtype` | Manifest | | `dtype` | Manifest |
| --- | --- | --- | --- | --- |
| `bool` | `bool` | | `bf16` | `bfloat16` |
| `u8` | `uint8` | | `u32` | `uint32` |
| `i8` | `int8` | | `i32` | `int32` |
| `f8e4m3` | `float8_e4m3fn` | | `f32` | `float32` |
| `f8e5m2` | `float8_e5m2` | | `u64` | `uint64` |
| `u16` | `uint16` | | `i64` | `int64` |
| `i16` | `int16` | | `f64` | `float64` |
| `f16` | `float16` | | | |

`init` is `zeros` (default), `ones`, `random` (with `seed`, a uint32, default 0) or `tensor`
(the tensor of the buffer's name in the tensors file, of the same type and shape).

### 3.2 Arguments

| `kind` | Fields | Manifest |
| --- | --- | --- |
| `buffer` | `buffer`, `offset_bytes` (multiple of 16) | `{index, kind: "buffer", buffer_id, byte_offset, access}` |
| `scalar` | `dtype` (`i8` `u8` `i16` `u16` `i32` `u32` `i64` `u64` `f32` `f64`), `value` | `{index, kind: "scalar", abi_type, value}`; an integer beyond 2^53 is a decimal string |
| `tensor_map` | `buffer`, `offset_bytes`, `data_type`, `global_dims`, `global_strides_bytes`, `box_dims`, `element_strides`, `interleave`, `swizzle`, `l2_promotion`, `oob_fill` | the same fields with `buffer_id`, `byte_offset` and the manifest's `data_type` names (`f16` → `float16`, `tf32` → `tfloat32`, `f32_ftz` → `float32_ftz`, …); `element_strides` defaults to all 1, the enumerations to `none` |
| `bytes` | `hex`, `relocations` `[{offset, buffer, offset_bytes}]` | `{index, kind: "bytes", hex, relocations: [{offset, buffer_id, byte_offset}]}` |

The fields of a tensor map are `cuTensorMapEncodeTiled`'s, innermost dimension first. The
harness checks the argument count and every argument's size against the parameter layout
of the inspection when one is cached or needed.

## 4. Calls

Every call carries `Authorization: Bearer <MARSCOMPUTE_API_KEY>` (an `mc_live_v1_...` key)
to `https://control.marscompute.ai`, or to `MARSCOMPUTE_API_URL` when it is set.

| Step | Call | Harness |
| --- | --- | --- |
| hardware | `GET /v1/xtrace/capabilities` | `xtrace hardware` |
| credit | `GET /v1/xtrace/credit` | `xtrace credit` |
| upload | `POST /v1/xtrace/artifacts/uploads` `{kind, name, size_bytes, sha256, content_type}` → grant; `PUT` the bytes to `upload_url`, or each range to its `parts[].upload_url` keeping each part's `ETag`; `POST /v1/xtrace/artifacts/{id}/complete` `{sha256, size_bytes, parts?}` | every command that uploads |
| inspect | `POST /v1/xtrace/inspections` `{schema_version: "marscompute.xtrace.inspection-request.v1", cubin_artifact_id, kernel_symbol?}`; poll `GET /v1/xtrace/inspections/{id}`; download `result` | `xtrace inspect`, `xtrace sass`, and `estimate` / `submit` when a site names a line |
| quote | `POST /v1/xtrace/quotes` with the estimate input | `xtrace estimate` |
| submit | `POST /v1/xtrace/jobs` with the estimate input, `quote_id` and `idempotency_key`, and the same key in `Idempotency-Key` | `xtrace submit` |
| state | `GET /v1/xtrace/jobs/{id}`, polled every `poll_after_milliseconds` | `xtrace job`, `--wait` |
| list | `GET /v1/xtrace/jobs?limit=&cursor=` | `xtrace jobs` |
| cancel | `POST /v1/xtrace/jobs/{id}/cancel` with `{}` | `xtrace cancel` |
| result | `GET /v1/xtrace/jobs/{id}/result` → `manifest` and `downloads` | `xtrace fetch` |

`estimate` and `submit` upload the cubin, the launch manifest (kind `launch_manifest`,
named `manifest.json`), and the tensors and reference files when the job has them. A
submission whose answer never arrived is sent again with the same idempotency key, which
never starts a second job. Inspections are free and limited per tenant to 4 at a time and
120 an hour.

## 5. Results

`xtrace fetch` checks every file against its download grant and the result manifest,
decompresses gzip-encoded files, and writes them by name next to `result-manifest.json`:

| File | Holds |
| --- | --- |
| `xtrace.json` | the trace report, read by `xtrace show` |
| `summary.json` | `quality` (`usable`, `partial`, `unusable`), `reason_codes`, `correctness`, record counts and per-stage summaries |
| `correctness.json` | each check's `status` and per-output comparison: against the reference, and each instrumented stage against the uninstrumented launch |
| `trace-plan.json`, `provenance.json`, `baseline.json`, `runner.log` | the plan that ran, the environment, the uninstrumented timings, the job's log |
| `range_profile.json`, `pc_sampling.json`, `pc_sampling_stats.json` | the profiler stages, when selected |
| `result-manifest.json` | the result manifest: status, quality, reason codes, correctness, files and environment |

## 6. Errors

A refusal answers `{"error": {"code", "message", "correlation_id"}}`. The harness prints
the code, the message and the correlation id.

| Code | Status | Meaning |
| --- | --- | --- |
| `insufficient_credit` | 402 | the credit cannot hold the quote; the error carries `top_up_url`, the console page where credit is added |
| `inspection_rate_limited` | 429 | too many inspections at once or this hour |
| `invalid_payload` | 400 | a body does not match its schema |

A failed inspection carries `failure.code`: `binary_invalid`, `arch_unsupported` or
`kernel_not_found`. A failed job carries `error` with `code`, `message`,
`classification` and `retryable`.

## 7. Cost

A quote freezes `limits.max_total_cost_micro_usd`, the most the job can cost, and
`rate_snapshot.gpu_micro_usd_per_hour`. Admission holds the cap against the tenant's
credit; the job settles at what it used. `xtrace submit` prints both and asks before it
submits, unless `--yes`.

## 8. Local MCP server

`xtrace mcp` serves the commands as MCP tools over stdio (JSON-RPC 2.0, one message per
line), on local paths: `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_inspect`, `xtrace_sass`, `xtrace_estimate`, `xtrace_submit`,
`xtrace_jobs`, `xtrace_job`, `xtrace_wait`, `xtrace_cancel`, `xtrace_fetch` and
`xtrace_show`. `xtrace_submit` takes `accept_max_cost_usd` and refuses a quote whose cap
exceeds it. `.mcp.json` registers the server for Claude Code.
