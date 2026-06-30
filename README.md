# Pretrain Healthcheck

`pretrain_healthcheck` 是训练前健康检查工具，用于在训练任务启动前或训练任务异常后，对已申请到的 pod / 节点组进行静态环境检查、单节点卡内检查、多节点集合通信检查和故障注入验证。

当前主线已经在两节点 MetaX C550 集群上完成验证，使用 `vcctl` 编排已创建的 vcjob pod。详细测试记录见：

```text
../design_docs/pretrain_healthcheck_metax_test_summary.md
```

## 1. 当前支持范围

已支持：

- 通过 `vcctl pod get` 获取 vcjob pod 信息；
- 通过 `vcctl pod exec` 在每个 pod 内执行健康检查；
- 测试前清理残留健康检查进程；
- pod 内静态环境探测；
- 单节点 8 卡动态检查；
- 多节点 `torchrun` 动态检查；
- smoke 快速连通性检查；
- GEMM、显存拷贝、集合通信、MoE / EP 类 `all_to_allv` 检查；
- NaN、显式 corrupt、hang / timeout、backend 初始化失败等故障注入；
- 外层 `summary.md` / `summary.json` / `results.jsonl` 汇总；
- rank 级和 group 级 JSONL 明细；
- Markdown 分析报告。

当前已验证硬件：

| 硬件 | 当前状态 |
| --- | --- |
| MetaX C550 | 已完成两节点 vcctl 流程验证，默认单节点 8 卡 |
| NVIDIA H200 | 保留单节点 6 卡脚本，未作为当前主流程 |

## 2. 目录结构

```text
pretrain_healthcheck/
  pretrain_healthcheck/
    cli.py                 # CLI 入口
    torch_checks.py        # 动态计算 / 通信检查
    analyze.py             # JSONL 结果分析和 report.md 生成
    static_checks.py       # Python 静态检查
  scripts/
    metax/
      run_vcctl_healthcheck.sh    # MetaX vcctl 主入口
      run_single_node_8c550.sh    # MetaX 单节点 8 卡检查
      probe_pod_capabilities.sh   # MetaX pod 静态能力探测
    nvidia/
      run_single_node_6h200.sh    # NVIDIA 单节点 6 卡检查
      dry_run_single_node_6h200.sh
      probe_pod_capabilities.sh
  tools/
    vcctl_healthcheck_driver.py   # vcctl 编排器
  results/
    vcctl/                        # vcctl 编排结果
```

## 3. 快速开始：MetaX vcctl 全流程

在开发机或可执行 `vcctl` 的环境中运行：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

DRY_RUN=0 \
MODE=all \
PROFILE=quick \
MESSAGE_SIZES=1M \
WARMUP=1 \
ITERS=1 \
bash run_vcctl_healthcheck.sh
```

当前 MetaX 脚本默认值面向 `muxi-2node`：

```text
JOB_NAME=muxi-2node
NAMESPACE=default
DEVICE_TYPE=metax
GPUS_PER_NODE=8
DIST_BACKEND=nccl
DEVICE_VENDOR=metax
COMM_RUNTIME=mccl
RESULT_ROOT=/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl
```

说明：

- `DIST_BACKEND=nccl` 是 PyTorch `torch.distributed` backend 名称；
- MetaX 环境底层通信 runtime 记录为 `mccl`；
- `HEALTHCHECK_MASTER_PORT=auto` 默认会在 master pod 内自动选择可用端口，避免复用 vcjob 默认 `MASTER_PORT=23456`。

## 4. 常用测试模式

### 4.1 smoke 快速连通性

用于快速确认开发机可以通过 `vcctl` 控制 pod，两个 pod 可以拉起 `torchrun`，并完成最小 `all_reduce`。

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

DRY_RUN=0 MODE=multi-node PROFILE=smoke bash run_vcctl_healthcheck.sh
```

特点：

- 每节点只启动 1 个 rank；
- 执行分布式初始化、小 tensor `all_reduce`、`all_gather_object` 和 `barrier`；
- rank0 写出 `ping_summary.json`。

### 4.2 正常多节点 quick

用于快速检查当前 vcjob pod 组的多节点通信、计算和正确性。

```bash
DRY_RUN=0 MODE=multi-node PROFILE=quick bash run_vcctl_healthcheck.sh
```

可缩短测试时间：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=quick \
MESSAGE_SIZES=1M \
WARMUP=1 \
ITERS=1 \
bash run_vcctl_healthcheck.sh
```

### 4.3 全流程验收

用于训练前完整验收当前 pod 组。

```bash
DRY_RUN=0 \
MODE=all \
PROFILE=quick \
MESSAGE_SIZES=1M \
WARMUP=1 \
ITERS=1 \
bash run_vcctl_healthcheck.sh
```

`MODE=all` 会依次执行：

```text
pre-clean
static
single-node
multi-node
```

## 5. MODE 与 PROFILE

| 参数 | 含义 |
| --- | --- |
| `MODE=static` | 只执行 pod 内静态环境探测 |
| `MODE=single-node` | 每个 pod 内单机 8 卡测试 |
| `MODE=multi-node` | 当前 vcjob 的多节点测试 |
| `MODE=all` | 依次执行 pre-clean、static、single-node、multi-node |
| `PROFILE=smoke` | 最短连通性测试，`nproc-per-node=1`，只做简单 `all_reduce` |
| `PROFILE=quick` | 快速正式测试，默认每节点 8 rank，覆盖计算、显存拷贝和通信算子 |

## 6. quick 覆盖算子

`PROFILE=quick` 当前覆盖：

```text
GEMM
memory_copy
all_reduce
reduce_scatter
all_gather
broadcast
all_to_all
all_to_allv
```

`all_to_allv` 用于覆盖 MoE / EP 类非均匀通信模式，当前 payload pattern 包括：

```text
uniform
skewed
hot_expert
random
empty_expert
```

## 7. 故障注入测试

故障注入用于验证健康检查工具自身是否能正确识别异常，并将异常传播到外层 `summary.md`。

### 7.1 NaN 精度异常

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=quick \
FAULT_NAN_RANK=3 \
bash run_vcctl_healthcheck.sh
```

预期：

- `overall_status=FAIL`；
- `rank_detail.jsonl` 中 rank3 出现 `rank_nan_count > 0`；
- `report.md` 中 `failed summaries > 0`。

### 7.2 显式 corrupt 数据异常

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=quick \
FAULT_CORRUPT_RANK=3 \
bash run_vcctl_healthcheck.sh
```

预期：

- `overall_status=FAIL`；
- `rank_detail.jsonl` 中 rank3 出现 `rank_error_type=FaultInjectedCorrupt`；
- `report.md` 中 `failed summaries > 0`。

### 7.3 hang / timeout

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=quick \
FAULT_SLEEP_RANK=3 \
FAULT_SLEEP_SECONDS=600 \
EXEC_TIMEOUT_SECONDS=120 \
bash run_vcctl_healthcheck.sh
```

预期：

- `overall_status=FAIL`；
- `results.jsonl` 中对应 pod 记录 `timeout=true`；
- `reason=timeout`；
- stderr 中包含 `TIMEOUT after 120s`。

### 7.4 后端初始化失败

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=quick \
FAULT_BACKEND=1 \
bash run_vcctl_healthcheck.sh
```

预期：

- `overall_status=FAIL`；
- stderr 中包含 `failed to initialize torch.distributed backend`；
- 该场景发生在 `init_process_group`，因此通常不会生成 rank/group 级算子明细。

## 8. 单节点测试

### 8.1 MetaX C550 单节点 8 卡

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck

bash scripts/metax/run_single_node_8c550.sh
```

常用参数：

```bash
GPUS_PER_NODE=8 \
DIST_BACKEND=nccl \
DEVICE_VENDOR=metax \
COMM_RUNTIME=mccl \
DTYPE=bf16 \
MESSAGE_SIZES=1M,16M,64M \
MOE_PATTERNS=uniform,skewed,hot_expert,random,empty_expert \
WARMUP=2 \
ITERS=5 \
bash scripts/metax/run_single_node_8c550.sh
```

### 8.2 NVIDIA H200 单节点 6 卡

NVIDIA 脚本保留用于单节点开发验证：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck

bash scripts/nvidia/dry_run_single_node_6h200.sh
bash scripts/nvidia/run_single_node_6h200.sh
```

说明：当前 README 主流程以 MetaX + vcctl 为准；NVIDIA 脚本没有参与当前两节点 vcctl 验证基线。

## 9. 静态环境探测

### 9.1 MetaX pod 能力探测

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck

bash scripts/metax/probe_pod_capabilities.sh
```

MetaX probe 会收集：

- `mx-smi` 输出；
- PyTorch 版本、设备可见性；
- `MACA` / `MCCL` / `MASTER_*` / `RANK` / `WORLD_SIZE` 环境变量；
- HCA / RDMA / IB 可见性；
- `xscale` / `net*` / `eth*` 网络接口信息；
- NUMA、磁盘、`/proc`、`/sys` 可见性。

### 9.2 dmesg 策略

当前运维提供的 pod 不支持读取 `dmesg`，因此 probe 脚本不在 pod 内执行 `dmesg`。summary 中记录为：

```text
logs | dmesg | SKIP | pod environment does not support dmesg; kernel log screening is owned by ops
```

宿主机内核日志、PCIe AER、MCE、soft lockup、XID / NPU 历史错误筛查由运维侧提供结论。

## 10. 输出目录与结果文件

vcctl 编排结果默认写入：

```text
${RESULT_ROOT}/${RUN_ID}/
```

MetaX 默认：

```text
/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl/<RUN_ID>/
```

主要文件：

| 文件 | 含义 |
| --- | --- |
| `summary.md` | 外层 vcctl Markdown 汇总 |
| `summary.json` | 外层 vcctl 机器可读汇总 |
| `results.jsonl` | 每个 pod / mode 的执行状态 |
| `commands.env` | 本次实际命令和关键环境变量 |
| `vcctl_pods.raw.json` | `vcctl pod get -o json` 原始输出 |
| `pods.jsonl` | 解析后的 pod 元信息 |
| `pods.tsv` | 便于查看的 pod 表 |
| `logs/<pod>.<mode>.stdout` | 每个 pod / mode 的 stdout |
| `logs/<pod>.<mode>.stderr` | 每个 pod / mode 的 stderr |
| `pod_results/<pod>/<mode>/` | 注入给检测程序的 pod 结果目录 |

动态测试的 pod 结果目录中常见文件：

| 文件 | 含义 |
| --- | --- |
| `report.md` | 分析器生成的 Markdown 报告 |
| `group_summary.jsonl` | group 级 op / message size / payload pattern 汇总 |
| `rank_detail.jsonl` | rank 级明细，用于定位慢 rank、NaN、Inf、错误 rank |
| `ping_summary.json` | smoke 模式的最小 all_reduce 连通性结果 |

## 11. 常用环境变量

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `JOB_NAME` | `muxi-2node` | vcjob 名称，用于 `vcctl pod get --job` |
| `NAMESPACE` | `default` | namespace |
| `MODE` | `all` | `static`、`single-node`、`multi-node`、`all` |
| `DEVICE_TYPE` | `metax` | 结果元信息，例如 `gpu`、`npu`、`metax` |
| `PROJECT_REMOTE_DIR` | `/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck` | pod 内项目路径 |
| `PROFILE` | `quick` | `quick` 或 `smoke` |
| `PRE_CLEAN` | `1` | 每次测试前清理可能残留的健康检查进程 |
| `GPUS_PER_NODE` | `8` | 每个 pod / 节点的 GPU 数 |
| `DIST_BACKEND` | `nccl` | PyTorch distributed backend 名称 |
| `HEALTHCHECK_MASTER_PORT` | `auto` | 健康检查 `torchrun` rendezvous 端口 |
| `DEVICE_VENDOR` | `metax` | 写入结果的硬件厂商元信息 |
| `COMM_RUNTIME` | `mccl` | 写入结果的底层通信 runtime 元信息 |
| `DTYPE` | `bf16` | 动态测试数据类型 |
| `MESSAGE_SIZES` | `1M,16M,64M` | 通信 payload 大小列表 |
| `MOE_PATTERNS` | `uniform,skewed,hot_expert,random,empty_expert` | MoE token routing 模式 |
| `WARMUP` | `2` | 每个测试项 warmup 次数 |
| `ITERS` | `5` | 每个测试项计时迭代次数 |
| `SEED` | `20260623` | 随机种子 |
| `RESULT_ROOT` | `/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl` | 共享结果根目录 |
| `RUN_ID` | 当前时间戳 | 本次运行 ID |
| `EXEC_TIMEOUT_SECONDS` | `3600` | 每个 pod exec 的超时时间 |
| `MAX_PARALLEL` | `0` | 最大并发 pod 数，`0` 表示全部并发 |
| `CONTAINER_NAME` | 空 | 强制指定 container；为空时自动选择 |
| `DRY_RUN` | `1` | `1` 表示只打印命令，不真正执行 |
| `POD_JSON_FILE` | 空 | 使用本地 pod JSON 文件做解析测试，不调用 `vcctl` |

故障注入变量：

| 环境变量 | 含义 |
| --- | --- |
| `FAULT_BACKEND=1` | 强制使用非法 backend，验证初始化失败采集 |
| `FAULT_SLEEP_RANK=<rank>` | 指定全局 rank 在测试中 sleep，模拟慢 rank / hang |
| `FAULT_SLEEP_SECONDS` | sleep 秒数，默认 `30` |
| `FAULT_NAN_RANK=<rank>` | 指定全局 rank 注入 NaN |
| `FAULT_CORRUPT_RANK=<rank>` | 指定全局 rank 修改 tensor，模拟结果污染 |

## 12. 已验证结果

当前最终基线：

```text
results/vcctl/20260630_111215
```

结果：

```text
overall_status: PASS
result_count: 8
pre-clean:   2/2 PASS
static:      2/2 PASS
single-node: 2/2 PASS
multi-node:  2/2 PASS
```

已验证异常链路：

| 测试项 | 代表目录 | 结果 |
| --- | --- | --- |
| NaN 注入 | `results/vcctl/20260630_103548` | 外层 FAIL，定位 rank3 NaN |
| corrupt 注入 | `results/vcctl/20260630_103811` | 外层 FAIL，定位 rank3 `FaultInjectedCorrupt` |
| timeout 注入 | `results/vcctl/20260630_105219` | 外层 FAIL，记录 timeout |
| backend 初始化失败 | `results/vcctl/20260630_105957` | 外层 FAIL，stderr 记录 backend 初始化错误 |

完整测试结果总结见：

```text
../design_docs/pretrain_healthcheck_metax_test_summary.md
```

## 13. 当前限制

- 当前 vcctl 多节点验证只在两节点 MetaX C550 环境完成闭环。
- 更大规模的 8 / 64 / 128 节点测试需要结合实际资源申请和分组策略继续验证。
- pod 内不执行 `dmesg`，宿主机内核日志筛查由运维侧提供。
- static probe 中部分系统命令可能在 pod 内缺失，例如 `rdma`、`ip`、`numactl`；这类结果用于能力探测，不等价于训练不可用。
- 当前 NPU / HCCL 完整执行路径尚未作为主流程验证。

## 14. 建议使用顺序

训练前建议按以下顺序执行：

1. `PROFILE=smoke MODE=multi-node`：快速确认 vcctl、torchrun、多 pod 连通性。
2. `MODE=all PROFILE=quick MESSAGE_SIZES=1M WARMUP=1 ITERS=1`：完整快速验收。
3. 若需要更充分压力，可增加 `MESSAGE_SIZES`、`WARMUP`、`ITERS`。
4. 若工具变更后需要自测，依次运行 NaN、corrupt、timeout、backend failure 故障注入。

常用命令：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

DRY_RUN=0 MODE=multi-node PROFILE=smoke bash run_vcctl_healthcheck.sh

DRY_RUN=0 MODE=all PROFILE=quick MESSAGE_SIZES=1M WARMUP=1 ITERS=1 bash run_vcctl_healthcheck.sh
```
