# Xtrace job protocol

This document is the contract between the harness and MarsCompute. It covers the job
directory, the job specification, the REST and MCP interfaces, and the result files.

## 1. The job

A job is data. It names a compiled kernel, how to launch it, what its buffers hold, and
which of its instructions to trace. The platform runs no program of the user's: it loads
the binary on a GPU of the chosen class, allocates and fills the buffers, and launches
the kernel itself.

A job directory holds a manifest, `xtrace.toml`, and the files it names:

| File | Holds |
| --- | --- |
| the binary | a cubin, or PTX that the driver compiles on the GPU |
| the sources | optional source files, shown beside the SASS listing |
| the tensors file | optional safetensors file with the contents of `init = "file"` buffers |

The **bundle** is a gzip'd tar of the manifest and those files, nothing else, at most
512 MiB. Every path stays inside the directory.

An **inspect** job reads the binary and returns the kernel's SASS listing and parameter
layout; it does not launch the kernel. A **trace** job launches the kernel once untraced
and once with the sites instrumented, restoring every buffer to its initial contents
before each launch, and returns the trace.

## 2. The manifest and the job specification

| Key | Meaning |
| --- | --- |
| `kind` | `inspect` or `trace` (default) |
| `hardware` | the hardware class, from `xtrace hardware` |
| `timeout_seconds` | the job's wall-clock limit on the GPU, 300 by default |
| `intent` | optional: what the user wants to learn, in their words, at most 2000 characters |
| `tensors` | optional: the safetensors file |
| `[kernel]` | `binary`, `name` (the entry point as the binary names it), `sources` |
| `[[buffers]]` | `name`, `dtype`, `shape`, `init` (`zeros`, `ones`, `random` or `file`), `output` |
| `[launch]` | `grid`, `block`, `cluster` (1 to 3 extents each), `shared_bytes` (dynamic shared memory), `args` |
| `[trace]` | the sites and regions, below |

Element types are `bool`, `u8`, `i8`, `f8e4m3`, `f8e5m2`, `u16`, `i16`, `f16`, `bf16`,
`u32`, `i32`, `f32`, `u64`, `i64` and `f64`. A `file` buffer takes the tensor of the same
name from the tensors file, with the same type and shape. An `output` buffer is returned
in `outputs.safetensors`, and the trace job checks that the traced launch wrote the same
bytes into it as the untraced one.

`launch.args` lists the kernel's parameters in order:

| `kind` | Fields | Passes |
| --- | --- | --- |
| `buffer` | `buffer`, `offset_bytes` | the buffer's device address plus the offset |
| `scalar` | `dtype`, `value` | the number, of an integer or float type |
| `tensor_map` | `buffer`, `offset_bytes`, `data_type`, `global_dims`, `global_strides_bytes`, `box_dims`, `element_strides`, `interleave`, `swizzle`, `l2_promotion`, `oob_fill` | a `CUtensorMap` the platform encodes with `cuTensorMapEncodeTiled` over the buffer; the fields are that call's parameters, innermost dimension first |
| `bytes` | `hex`, `relocations` | a by-value struct; each relocation writes a buffer's address (`buffer`, `offset_bytes`) at an 8-byte aligned `offset` |

The tensor map's enumerations read `data_type`: `u8` `u16` `u32` `i32` `u64` `i64` `f16`
`f32` `f64` `bf16` `f32_ftz` `tf32` `tf32_ftz`; `interleave`: `none` `16b` `32b`;
`swizzle`: `none` `32b` `64b` `128b`; `l2_promotion`: `none` `64b` `128b` `256b`;
`oob_fill`: `none` `nan`.

A trace job's `[trace]` table:

| Key | Meaning |
| --- | --- |
| `clock` | `gpu` (device-wide nanosecond timer, default), `sm` (per-SM cycle counter) or `anchor` (cycle counter rebuilt to nanoseconds; warp-level sites only) |
| `per_thread_records` | records each thread keeps before its ring wraps |
| `schedule` | `first_feasible` (default) or `least_makespan`, how carefully probes are placed |
| `[[trace.sites]]` | the instructions that record, in order; a site's position is its id |
| `[[trace.regions]]` | pairs of sites that bound an interval: `enter`, `leave`, `label` |
| `[trace.scope_roles]` | region label to the warp role that executes it, for the report's lanes |

A site names one instruction by `index` (from `xtrace sass`) or by `line`
(`<file>:<line>`, the first instruction emitted for that line; the binary needs line
info). It may record a value beside its timestamp: `value_register` (`R<n>` or `UR<n>`),
`value_width_bits` (32 or 64) or `value_is_smid`. `warp_level = true` writes one record
per warp instead of one per thread.

`xtrace upload` and `xtrace submit` turn the manifest into the JSON specification the
interfaces take; `examples/matmul` gives:

```json
{
  "schema": "xtrace.job/1",
  "kind": "trace",
  "kernel": {"binary": "matmul.cubin", "name": "square_matrix_mul", "sources": ["matmul.cu"]},
  "tensors": null,
  "buffers": [
    {"name": "a", "dtype": "i32", "shape": [256, 256], "init": "random", "output": false},
    {"name": "b", "dtype": "i32", "shape": [256, 256], "init": "random", "output": false},
    {"name": "c", "dtype": "i32", "shape": [256, 256], "init": "zeros", "output": true}
  ],
  "launch": {
    "grid": [16, 16, 1], "block": [16, 16, 1], "cluster": null, "shared_bytes": 0,
    "args": [
      {"kind": "buffer", "buffer": "a", "offset_bytes": 0},
      {"kind": "buffer", "buffer": "b", "offset_bytes": 0},
      {"kind": "buffer", "buffer": "c", "offset_bytes": 0},
      {"kind": "scalar", "dtype": "i32", "value": 256}
    ]
  },
  "timeout_seconds": 120,
  "intent": "how long each thread spends in the dot-product loop, and how evenly the blocks run",
  "trace": {
    "clock": "gpu",
    "per_thread_records": null,
    "schedule": "first_feasible",
    "sites": [
      {"line": "matmul.cu:8", "value_width_bits": 32, "value_is_smid": false, "warp_level": false},
      {"line": "matmul.cu:11", "value_width_bits": 32, "value_is_smid": false, "warp_level": false}
    ],
    "regions": [[0, 1, "dot product"]],
    "scope_roles": {}
  }
}
```

## 3. REST interface

Every request carries `Authorization: Bearer <MarsCompute API key>`. Errors answer
`{"error": {"code", "message"}}` with the request's `x-request-id`.

| Method and path | Does |
| --- | --- |
| `GET /v1/xtrace/hardware` | the hardware classes: `id`, `gpu`, `arch`, `gpu_second_micro_usd`, `job_micro_usd`, `timeout_seconds_max`, `available` |
| `GET /v1/xtrace/balance` | the tenant's available credit, `available_micro_usd` |
| `POST /v1/xtrace/uploads` | `{size_bytes, sha256}` grants an upload: `id`, `upload_url` (signed PUT), `expires_at` |
| `POST /v1/xtrace/uploads/{id}/complete` | checks the uploaded bundle against its size and digest |
| `POST /v1/xtrace/jobs` | `{upload_id, hardware, spec}` with an `Idempotency-Key` admits a job and reserves its maximum cost |
| `GET /v1/xtrace/jobs` | the account's latest jobs, `?limit=` |
| `GET /v1/xtrace/jobs/{id}` | one job |
| `POST /v1/xtrace/jobs/{id}/cancel` | asks a job to stop |

A job reads:

```json
{
  "id": "xtj_...",
  "kind": "trace",
  "state": "succeeded",
  "hardware": "h100-sxm",
  "created_at": "...", "started_at": "...", "finished_at": "...",
  "gpu_seconds": 4.2,
  "reserved_micro_usd": 0, "charge_micro_usd": 0,
  "error": null,
  "summary": {},
  "files": [{"name": "trace.json", "size_bytes": 150393, "url": "https://..."}]
}
```

`state` is one of `queued`, `running`, `cancel_requested`, `succeeded`, `failed`,
`cancelled` and `expired`. A failed job's `error.code` is one of `binary_invalid`,
`kernel_not_found`, `arch_unsupported`, `launch_invalid`, `launch_failed`,
`site_unresolved`, `site_not_instrumentable`, `timeout` and `platform_failure`. A file's
`url` is signed and valid for five minutes; fetch the job again for a fresh one.

## 4. MCP interface

The MCP server is at `https://api.marscompute.ai/mcp` (Streamable HTTP, the same bearer
key). Its tools mirror the REST interface, except for moving bytes, which a tool call
cannot carry:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `xtrace_list_hardware` | none | the hardware classes |
| `xtrace_get_balance` | none | the available credit |
| `xtrace_submit_job` | `upload_id`, `hardware`, `spec` | the admitted job |
| `xtrace_get_job` | `job_id` | the job, with its summary and file links |
| `xtrace_list_jobs` | `limit` | the latest jobs |
| `xtrace_cancel_job` | `job_id` | the job |

`xtrace upload <directory>` uploads the bundle and prints the `upload_id` and `spec` to
pass to `xtrace_submit_job`; `xtrace fetch <job>` downloads the files `xtrace_get_job`
links to.

## 5. Result files

| File | Job | Holds |
| --- | --- | --- |
| `inspect.json` | inspect | `xtrace.inspect/1`: the binary's kernels, and the named kernel's registers, static shared memory, parameters (index, offset, size) and instructions (index, pc, SASS text, source line, basic block) |
| `trace.json` | trace | the trace report, read by `xtrace show` |
| `report.html` | trace | the same report as an interactive page |
| `summary.json` | trace | the headline figures `xtrace_get_job` returns, with `outputs_match` |
| `outputs.safetensors` | trace | the `output` buffers after the untraced launch, at most 256 MiB |
| `log.txt` | both | the job's messages: what was loaded and launched, warnings, the cause of a failure |

## 6. Cost

A trace job is charged a fixed price per job plus a price per GPU-second of the time the
kernel's launches held the GPU, at the prices `xtrace hardware` lists when the job is
admitted. An inspect job is charged the fixed price only. Admission reserves the price
of the whole timeout against the tenant's credit and refuses the job with
`insufficient_credit` when the credit does not cover it. A job that fails because of the
platform is not charged; one whose launch fails, or that reaches its timeout, is charged
for the time it ran.
