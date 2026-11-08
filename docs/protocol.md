# From xtrace.toml to the Xtrace API

The harness turns a job directory into calls to the Xtrace API of the MarsCompute Control
Plane. This document maps every key of `xtrace.toml` onto that API and lists the calls in
the order the harness makes them.

## 1. Source of truth

The platform's contracts define the API: the zod schemas in `packages/contracts/src/`
(`xtrace.ts`, `xtrace-inspection.ts`, `xtrace-launch-manifest.ts`) of
mars-compute-ai/marscompute-maas, exported as JSON Schema to
`packages/contracts/json-schema/xtrace/*.schema.json`. Agent mode is specified in
`docs/reference/xtrace-agent.md` of the same repository. Where this document and those
sources disagree, the sources win.

The harness vendors the exported schemas into `xtrace_harness/schemas/`:

```bash
python scripts/sync_schemas.py <maas checkout>           # copy them
python scripts/sync_schemas.py <maas checkout> --check   # exit 1 when the copy is stale
```

The tests validate every request body the client builds against the vendored schema of
its route (`upload-begin`, `upload-complete`, `inspection-request`, `inspection-accept`,
`inspection-downloads-request`, `estimate-input`, `submit-input`), every launch manifest it
uploads against `launch-manifest`, and the fake Control Plane's capabilities, quotes and
inspection documents against theirs. A schema that is not vendored is skipped.

## 2. The job directory

A job directory holds `xtrace.toml` and the files it names. Every path stays inside the
directory. A job is one of two kinds:

- **A launch job** describes one launch of one kernel of a cubin: its shape, its arguments
  and its buffers' contents, and what to trace (`[trace]`) or a request for the platform
  to plan the trace (`[planner]`). It uploads data only; no program of the user's runs.
- **An agent-mode job** (`[workspace]` and `[agent]`) hands a workspace to an agent on the
  GPU host: cubins, and optionally a source tree, a launch manifest, tensors and other
  files. The agent may read, build and run what the workspace holds, within its budget,
  and returns one `xtrace.json`.

| File | Holds | Uploaded as |
| --- | --- | --- |
| `xtrace.toml` | the manifest | translated, sections 3 and 4 |
| the cubin | the kernel; PTX is not accepted | artifact `cubin` |
| the sources of `[kernel] sources` | optional; the viewer shows them beside the listing | not uploaded |
| the tensors file | optional safetensors with the contents of `init = "tensor"` buffers | artifact `safetensors`, see section 3.1 |
| the reference file | optional safetensors with the expected contents of the `[validation]` output | artifact `reference` |
| `[workspace]` files | agent mode, section 4 | artifacts `cubin`, `source`, `launch_manifest`, `safetensors`, `workspace_file` |

`.xtrace-cache/` beside a cubin holds its inspections, keyed by the cubin's sha256 and the
kernel symbol (section 5).

## 3. Launch jobs

| `xtrace.toml` | API |
| --- | --- |
| `hardware = "<environment_id>"` | `compute` = the `compute` of the capabilities entry with that `environment_id`, which must list `cubin_launch` in `bundle_kinds`; its `architecture` becomes `target.architecture` |
| `intent` (required) | `prompt` |
| `[kernel] binary` | artifact kind `cubin` → `bundle.cubin_artifact_id` |
| `[kernel] name` | `target.kernel_symbol` |
| `[kernel] sources` | not uploaded |
| `tensors` | artifact kind `safetensors` → `bundle.tensors_artifact_id` |
| `reference` + `[validation]` (`output`, `atol`, `rtol`) | artifact kind `reference` → `bundle.reference_artifact_id`; manifest `validation` `{output_key, reference_key, atol, rtol}`, both keys the output's name |
| `[[buffers]] name, dtype, shape` | manifest `allocations[]` `{id, size_bytes, init}` and `buffers[]` `{name, allocation_id, offset_bytes: 0, size_bytes, access}`, one view over the whole allocation |
| `[[buffers]] init` | `zeros` → `init.kind = "zeros"`; `tensor`, `ones`, `random` → `init = {kind: "tensor", tensor_key: <name>}`, section 3.1 |
| `[[buffers]] output = true` | the view's `access = "read_write"` (else `"read"`), and an entry of `outputs[]` `{name, buffer, dtype, shape}` |
| `[launch] grid, block, cluster, shared_bytes` | `target.grid`, `target.block`, `target.cluster` (`null` without a cluster), `target.dynamic_shared_memory_bytes` |
| `[launch] args` | `arguments[]` with `index` = position |
| a cached inspection of the kernel | `parameter_layout` `[{index, offset, size_bytes}]` when it extracted the layout, else `null` |
| `[[trace.sites]] index` | `sites[].instruction_index`: the instruction's index in `xtrace sass` |
| `value_register`, `value_width_bits` | `probe = "value"`, `value_register`, `value_width_bits`; `capture_values = true` |
| `value_is_smid` | `probe = "smid"`; `capture_values = true` |
| `warp_level` (alike on every site) | `granularity` `"warp"`, else `"thread"` |
| `per_thread_records` (default 32) | `max_records` = records × rows; rows are the launch's threads, or its warps |
| `[trace] clock`, `schedule`, `[trace.scope_roles]` | `clock`, `schedule`, `scope_roles` |
| `[[trace.regions]] enter, leave, label` | `regions[]` `{region_id = label, start_site_id = "s<enter>", end_site_id = "s<leave>"}` |
| `[trace] stages` with `[trace.range_profile]`, `[trace.pc_sampling]` | `stages`, `range_profile`, `pc_sampling` |
| `[planner] model, max_calls, max_input_tokens, max_output_tokens` (instead of `[trace]`) | `analysis = {mode: "platform_planned", model, max_calls, max_input_tokens_total, max_output_tokens_total, capture_values: false}` |
| `[limits] max_cost_usd, max_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes` | `limits` `{max_total_cost_micro_usd, max_active_gpu_seconds, max_wall_seconds, max_attempts, max_trace_bytes}`; an absent key takes the capabilities' `limits` |

The launch manifest is `marscompute.kernel-launch`, uploaded as kind `launch_manifest`
named `manifest.json`, with `runtime_profile: null` and `reset.policy =
"restore_pristine_before_every_invocation"`. Its `target.architecture` is one of `sm_80`,
`sm_86`, `sm_89`, `sm_90`, `sm_90a`, `sm_100`, `sm_100a`, `sm_103` and `sm_103a`; the
harness refuses a launch job on any other architecture.

Site `n` is the n-th `[[trace.sites]]` entry and has `site_id = "s<n>"`. A region label is
1 to 200 letters, digits, `_`, `.`, `:` or `-`; an unlabelled region is `s<enter>-s<leave>`.
`[trace]` with sites becomes `analysis = {mode: "explicit_plan", trace_plan}` with
`tier = "sass"`. When an inspection of the kernel is cached, every site's index must be an
instruction of its listing and the arguments must match its parameter layout.

### 3.1 Buffers and tensors

| `dtype` | Output dtype | | `dtype` | Output dtype |
| --- | --- | --- | --- | --- |
| `bool` | `bool` | | `bf16` | `bfloat16` |
| `u8` | `uint8` | | `u32` | `uint32` |
| `i8` | `int8` | | `i32` | `int32` |
| `f8e4m3` | `uint8` | | `f32` | `float32` |
| `f8e5m2` | `uint8` | | `u64` | `uint64` |
| `u16` | `uint16` | | `i64` | `int64` |
| `i16` | `int16` | | `f64` | `float64` |
| `f16` | `float16` | | | |

An `f8` output is compared byte for byte, so it cannot be the `[validation]` output. A
buffer name is 1 to 128 letters, digits and `_`, and does not start with `_xtrace`.

`init` is `zeros` (default), `ones`, `random` (with `seed`, a uint32, default 0) or `tensor`
(the tensor of the buffer's name in the tensors file, of the same type and shape). A launch
starts every non-zero buffer from a tensor, so the harness prepares `ones` and `random`
buffers itself:

- `ones` holds the value one in the buffer's type;
- `random` holds the bytes of a Mersenne Twister (Python's `random.Random(seed)`); a
  floating-point element keeps its random sign and mantissa and takes the exponent of a
  magnitude in [0.5, 1), a `bool` keeps the lowest bit, and an integer keeps every bit;
- the harness writes them, with the `init = "tensor"` tensors of the tensors file, into one
  safetensors file that it uploads as the job's tensors. Without prepared buffers it
  uploads the tensors file as it is.

### 3.2 Arguments

| `kind` | Fields | Manifest |
| --- | --- | --- |
| `buffer` | `buffer`, `offset_bytes` (multiple of 16) | `{index, kind: "buffer", buffer, offset_bytes}` |
| `scalar` | `dtype` (`i8` `u8` `i16` `u16` `i32` `u32` `i64` `u64` `f32` `f64`), `value` | `{index, kind: "scalar", abi_type, value}`, the value as a decimal string |
| `tensor_map` | `buffer`, `offset_bytes`, `data_type`, `global_dims`, `global_strides_bytes`, `box_dims`, `element_strides`, `interleave`, `swizzle`, `l2_promotion`, `oob_fill` | the same fields; `element_strides` defaults to all 1, the enumerations to `none` |
| `bytes` | `hex`, `relocations` `[{offset, buffer, offset_bytes}]` | `{index, kind: "bytes", hex, relocations: [{offset, buffer, offset_bytes}]}` |

The fields of a tensor map are `cuTensorMapEncodeTiled`'s, innermost dimension first, and
the map must lie inside its buffer from `offset_bytes`. A `bytes` argument holds 1 to 4096
bytes, with zeros at every relocation offset, where the platform writes the address. The
arguments occupy at most 4096 bytes together and carry at most 128 relocations.

## 4. Agent-mode jobs

| `xtrace.toml` | API |
| --- | --- |
| `hardware = "<environment_id>"` | `compute` of that capabilities entry, which must list `agent_workspace` in `bundle_kinds` and an architecture every cubin targets |
| `intent` (required) | `prompt` |
| `[workspace] cubins` (1 to 8) | artifacts kind `cubin` → `bundle.cubin_artifact_ids` |
| `[workspace] source` (a directory) | archived as `source.tar.gz` without `.git` → kind `source` → `bundle.source_archive_artifact_id` |
| `[workspace] manifest` (a `marscompute.kernel-launch` file) | kind `launch_manifest` → `bundle.launch_manifest_artifact_id` |
| `[workspace] tensors` (0 to 8 safetensors files) | kind `safetensors` → `bundle.tensor_artifact_ids` |
| `[workspace] files` (0 to 8 files) | kind `workspace_file` → `bundle.file_artifact_ids` |
| `[agent] model` | `analysis.model`, one of the capabilities' `agent_models` |
| `[agent] max_turns` (1 to 500), `max_input_tokens` (1 to 50,000,000), `max_output_tokens` (1 to 1,000,000), `max_wall_seconds` (60 to 2580) | `analysis` `{mode: "agent", max_turns, max_input_tokens_total, max_output_tokens_total, max_wall_seconds}` |
| `[limits]` | `limits`, as for a launch job |

`bundle.kind` is `agent_workspace`. A file is named once in `[workspace]`. The source
archive is a gzip-compressed tar of the directory's files and subdirectories, paths
relative to it, with owners, times and modes normalized, so the same tree gives the same
archive; it leaves out every `.git`, `.xtrace-cache` and `xtrace-results` entry. An
agent-mode job takes neither `[kernel]`, `[launch]`, `[[buffers]]`, `[trace]` nor
`[planner]`, and needs no inspection.

## 5. Inspections

An inspection reads a cubin's kernel catalog statically: each kernel's architecture and
parameter layout, and a listing of its instructions. It is quoted and charged a fixed fee
per inspection, which the user accepts before it runs.

1. `GET /v1/xtrace/inspections/capabilities` lists the inspection profiles with their
   limits and fixed fee, or why inspections are unavailable.
2. The cubin is uploaded (kind `cubin`) and `POST /v1/xtrace/inspections/quotes` takes
   `{schema_version: "marscompute.xtrace.inspection-request.v1", binary: {artifact_id,
   sha256, size_bytes, format: "cubin"}, kernel_selector, inspection_profile_id}`;
   `kernel_selector` is `{kind: "symbol", symbol}` with `--kernel`, else `{kind: "all"}`.
3. `POST /v1/xtrace/inspections` accepts the quote:
   `{schema_version: "marscompute.xtrace.inspection-accept.v1", quote_id, quote_sha256,
   idempotency_key, request}`, with the key also in `Idempotency-Key`. `quote_sha256` is
   the SHA-256 of the quote's canonical JSON: keys sorted, no whitespace.
4. `GET /v1/xtrace/inspections/{id}` is polled until `execution.state` is `terminal`; its
   `outcome` is `completed`, `rejected_input`, `timed_out`, `cancelled` or
   `platform_failed`.
5. `GET /v1/xtrace/inspections/{id}/result` returns the result,
   `marscompute.xtrace.inspection-result.v1`. The harness checks that it describes the
   cubin's sha256.
6. `POST /v1/xtrace/inspections/{id}/artifacts/downloads` `{artifact_ids}` grants the
   kernel's instruction listing, `marscompute.xtrace.static-instructions.v1`, which the
   harness checks against the result's digest and size.

The harness caches the result as `<sha256>.<kernel>.inspection-result.json` and the
listing beside it as `<sha256>.<kernel>.instructions.json`. A listed instruction's index
is its `function_index`, the value a site's `index` names. `GET
/v1/xtrace/inspections?limit=&cursor=` pages the account's inspections and `POST
/v1/xtrace/inspections/{id}/cancel` with `{}` asks one to stop.

## 6. Calls

Every call carries `Authorization: Bearer <MARSCOMPUTE_API_KEY>` (an `mc_live_v1_...` key)
to `https://control.marscompute.ai`, or to `MARSCOMPUTE_API_URL` when it is set.

| Step | Call | Harness |
| --- | --- | --- |
| hardware | `GET /v1/xtrace/capabilities` | `xtrace hardware` |
| credit | `GET /v1/xtrace/credit` | `xtrace credit` |
| upload | `POST /v1/xtrace/artifacts/uploads` `{kind, name, size_bytes, sha256, content_type}` → grant; `PUT` the bytes to `upload_url`, or each range to its `parts[].upload_url` keeping each part's `ETag`; `POST /v1/xtrace/artifacts/{id}/complete` `{sha256, size_bytes, parts?}` | every command that uploads |
| inspect | section 5 | `xtrace inspect`, `xtrace inspection ...` |
| quote | `POST /v1/xtrace/quotes` with the estimate input | `xtrace estimate` |
| submit | `POST /v1/xtrace/jobs` with the estimate input, `quote_id` and `idempotency_key`, and the same key in `Idempotency-Key` | `xtrace submit` |
| state | `GET /v1/xtrace/jobs/{id}`, polled every `poll_after_milliseconds` | `xtrace job`, `--wait` |
| list | `GET /v1/xtrace/jobs?limit=&cursor=` | `xtrace jobs` |
| cancel | `POST /v1/xtrace/jobs/{id}/cancel` with `{}` | `xtrace cancel` |
| result | `GET /v1/xtrace/jobs/{id}/result` → `manifest` and `downloads` | `xtrace fetch` |

`estimate` and `submit` upload the job's files (sections 3 and 4) before they quote. A
submission or an inspection acceptance whose answer never arrived is sent again with the
same idempotency key, which never starts a second job or inspection.

## 7. Results

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

An agent-mode job returns `xtrace.json` and `result-manifest.json` only; a session that
ends without a trace has `quality: "unusable"`.

## 8. Errors

A refusal answers `{"error": {"code", "message", "correlation_id"}}`. The harness prints
the code, the message and the correlation id.

| Code | Status | Meaning |
| --- | --- | --- |
| `insufficient_credit` | 402 for a job, 409 for an inspection | the credit cannot hold the quote; the error carries `top_up_url`, the console page where credit is added |
| `inspection_result_not_ready` | 409 | the inspection has not completed |
| `invalid_payload` | 400 | a body does not match its schema |

A failed job carries `error` with `code`, `message`, `classification` and `retryable`.

## 9. Cost

A job quote freezes `limits.max_total_cost_micro_usd`, the most the job can cost, and
`rate_snapshot.gpu_micro_usd_per_hour`. In agent mode the quote's
`rate_snapshot.price_book` adds the job fee (`tool_fee_micro_usd`) and the agent model's
token rates per million input, cached input and output tokens; the cap must cover the fee,
the tokens at their caps and GPU time at `max_active_gpu_seconds`, and the GPU time
includes the time the agent thinks. Admission holds the cap against the tenant's credit;
the job settles at what it used. An inspection quote freezes its fixed fee
(`price.fixed_fee_micro_usd`). `xtrace inspect` and `xtrace submit` print the fee or the
cap and ask before they spend, unless `--yes`.

## 10. Local MCP server

`xtrace mcp` serves the commands as MCP tools over stdio (JSON-RPC 2.0, one message per
line), on local paths: `xtrace_hardware`, `xtrace_credit`, `xtrace_cubin_find`,
`xtrace_cubin_show`, `xtrace_inspection_capabilities`, `xtrace_inspection_estimate`,
`xtrace_inspection_submit`, `xtrace_inspection_get`, `xtrace_inspection_wait`,
`xtrace_inspection_query`, `xtrace_inspection_cancel`, `xtrace_inspection_result`,
`xtrace_inspection_downloads`, `xtrace_sass`, `xtrace_estimate`, `xtrace_submit`,
`xtrace_jobs`, `xtrace_job`, `xtrace_wait`, `xtrace_cancel`, `xtrace_fetch` and
`xtrace_show`. `xtrace_inspection_submit` takes `accept_max_fee_usd` and `xtrace_submit`
takes `accept_max_cost_usd`; each refuses a quote above it. `.mcp.json` registers the
server for Claude Code.
