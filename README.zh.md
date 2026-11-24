# MarsTrace

MarsTrace 在云端 GPU 上测量 CUDA kernel 内部的耗时。你提交 kernel 的 cubin，GPU 机器上的 agent 负责插桩，按你写的 launch 运行，然后返回 `marstrace.json`。结果里有每个 thread 或 warp 在各个区间的耗时、grid 的负载平衡，以及时间花在哪里的分析。本仓库是源码公开（source-available）的客户端，也就是 `marstrace` 命令和它的 MCP 服务，面向 kernel 开发者和他们的 coding agent，本地不需要 GPU。

[English](README.md)

## 快速上手

1. 在 [MarsCompute console](https://console.marscompute.ai/register) 注册，充值，然后在 MarsTrace › Keys 创建 API key。
2. 安装客户端，跑一下仓库自带的例子：

```bash
pip install 'git+https://github.com/mars-compute-ai/MarsTrace@v0.2.0'
export MARSTRACE_API_KEY=mt_live_...

marstrace estimate examples/matmul        # token 单价、这次 job 最多花多少、余额
marstrace submit examples/matmul --wait   # 提交、等待、下载，并打印分析结果
marstrace show examples/matmul/marstrace-results/<job>/marstrace.json --stats
```

一个 job 目录里放 `marstrace.toml`，以及它引用的文件：

```toml
hardware = "h200"   # 或 "b200"；cubin 要按对应架构编译（sm_90a 或 sm_100a）
intent = """
How long each thread spends in the dot-product loop of square_matrix_mul, and how evenly the
blocks run, at n = 256: int32 matrices a, b and c of 256 x 256, a and b random, launched with
grid (16, 16) and block (16, 16).
"""

[workspace]
cubins = ["matmul.cubin"]   # nvcc -cubin -arch=sm_90a -lineinfo -o matmul.cubin src/matmul.cu
source = "src"              # 可选；有源码时，结果能对应到源码行
```

`intent` 里要写清三件事：想回答的问题、问题规模、launch 的 grid 和 block。agent 会严格按这个 launch 来测。Triton、CuTeDSL、TileLang 等编出来的 kernel，可以用 `marstrace cubin find <kernel> --copy-to my-job` 从编译缓存里取出 cubin。`marstrace.toml` 的全部字段见 [docs/protocol.md](docs/protocol.md)。

## 能回答什么

每当一个 thread 或 warp 执行到你关心的指令，MarsTrace 就记一个时间戳，所以它测的是 kernel 内部两个点之间的时间：

- 每个循环、阶段或 warp 角色花了多长时间（按 thread 或按 warp），以及耗时的分布；
- block 和 SM 之间负载是否均匀，一个 SM 上同时跑几个 block，grid 末尾的拖尾有多长；
- producer 和 consumer warp 在哪里互相等待，以及各区间之间的空档。

它测不到 DRAM 和 L2 流量、访存合并、stall 原因、bank conflict、tensor pipe 的使用情况。分析结果会为每一项写明，用哪个 profiler 指标可以确认。它也不会修改你的 kernel。

## 怎么读结果

`marstrace fetch` 和 `submit --wait` 会在 `marstrace-results/<job>/` 下保存两个文件：

- `result-manifest.json`：先看 `quality`。
  - `usable` 表示所有检查都通过；
  - `partial` 时，`reason_codes` 会说明哪些数字不可靠；
  - `unusable` 表示没有得到时间线，这样的 job 不收费。
- `marstrace.json`：trace 本身和分析结果。trace 用 `marstrace show` 查看；分析结果 `marstrace fetch` 下载后会直接打印，`marstrace show <file> --analysis` 可以再看一次。

分析结果由一个确定性的分析器生成，不是模型写的，包括：

- **Findings**：每条都附有本次运行的证据和下一步建议，以及 **Where**，即所在的区间和区间两端 site 的 SASS 下标、opcode 和源码行。
- **假设**（列在 finding 下面）和**这次 trace 测不到的项目**：都是计时 trace 看不到的原因，每一项都写了用什么测量可以确认。在测量之前，假设不能当作原因。
- **平衡**：SM 利用率、拖尾、每个 SM 平均驻留的 block 数、trace 覆盖的时长，以及不插桩时的 kernel 时间。

## Agent：Claude Code 插件和 MCP

插件会装上 `marstrace mcp` 服务和两个 skill：`marstrace-remote` 负责准备和提交 job，`marstrace-read` 负责读结果。

```bash
claude plugin marketplace add mars-compute-ai/MarsTrace
claude plugin install marstrace@marscompute
```

其他 agent 可以把 `marstrace mcp` 配置成 stdio MCP 服务，并在环境变量里设好 `MARSTRACE_API_KEY`，写法参照 [.mcp.json](.mcp.json)。skill 在 [marstrace/skills/](marstrace/skills/) 下。

## 目前支持的范围

- 模式：fast，即 trace 加分析。用来优化 kernel 的 `thinking-light` 和 `thinking-heavy` 还未开放。
- GPU：H200、B200。
- CUDA：13.0。
- 输入：最多 8 个 cubin（必须是 ELF cubin，不接受 PTX）。另外可以附源码、launch 描述文件、safetensors 输入和其他文件，合计不超过 256 MiB。

## 计费

按 agent 在所有 attempt 中实际使用的 token 收费，单价以 `marstrace hardware` 和 `marstrace estimate` 显示的为准。GPU 时间不另外收费。以下两种 job 不收费：交付的 trace 判为 `unusable`，或者 job 失败、没有交付结果。job 被取消时，按取消前已经用掉的 token 收费。余额达到 `marstrace credit` 显示的最低额度，job 才会开始运行；`marstrace estimate` 会显示这次 job 最多可能花多少钱。

## 隐私与安全

- 每次 attempt 都在一台独立的 GPU 机器上运行，attempt 结束后机器即被删除，不同 job、不同客户之间不共享任何东西。
- 机器不接受任何入站连接，agent 所在的容器只能连到 MarsTrace 服务。
- 你上传的文件在 job 结束 24 小时后删除，结果可以在 30 天内下载。
- MarsCompute 会保留每个 job 中 agent 与模型往来的请求和回复，其中可能引用你的代码。这些记录在 job 结束后保留 30 天，到期自动删除；只有 MarsCompute 的管理员能查看。你提出要求时，可以提前删除。

## 常见问题

**本地需要 GPU 或 CUDA 吗？** 不需要。只有编译 cubin 时需要 `nvcc`，或者用编译缓存里现成的 cubin。

**支持哪些 kernel？** 只要能编成对应 GPU 架构的 cubin 就行，CUDA C++、Triton、CuTeDSL、TileLang、CUTLASS 都可以。

**kernel 的 launch 比较特殊怎么办？** 在 `intent` 里写清楚，或者在 `[workspace]` 里附上 launch 描述文件或驱动脚本。格式见 [docs/protocol.md](docs/protocol.md)。

**trace 的时长为什么和 kernel 时间不一样？** trace 只覆盖 agent 放下的各个 site 之间的时间。第一个 site 之前和最后一个 site 之后的工作，以及 launch 开销，都不在里面。分析结果会报告这部分差距。

## 参与贡献

欢迎提 issue 和 pull request。仓库结构、测试方法和改动要遵守的规则见 [AGENT.md](AGENT.md)。提 pull request 前请先跑 `python -m unittest discover -s tests`。

## 许可证

MarsTrace 客户端的源码按 [PolyForm Noncommercial License 1.0.0](LICENSE) 公开（source-available），不是 OSI 意义上的开源软件。你可以出于任何非商业目的使用、修改和分享它。[NOTICE](NOTICE) 里有必须保留的声明。本许可不允许商业使用。
