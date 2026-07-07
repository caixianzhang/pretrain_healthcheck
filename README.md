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
- 静态检查 compact 聚合输出，默认按 run 保存 JSONL，避免大规模节点产生大量小文件；
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

### 3.1 执行前预演 / 前置检查模式

`DRY_RUN=1` 是执行前预演模式，用于在真正进入 pod 执行检测前，先确认 `vcctl` 编排信息和下发命令是否正确。它不是临时 debug 功能。

预演模式会完成：

- 读取 `JOB_NAME` 对应的 pod 列表；
- 解析 master / worker、rank、world size、container name；
- 输出 pod 到物理节点的映射关系；
- 生成本次 `RUN_ID` 和结果目录；
- 打印将下发到每个 pod 的 `vcctl pod exec` 命令；
- 不会真正执行 pod 内检测程序。

大规模节点验收前建议先执行一次预演：

```bash
DRY_RUN=1 \
JOB_NAME=<vcjob-name> \
MODE=static \
bash run_vcctl_healthcheck.sh
```

确认 pod 数量、节点映射、rank 顺序和命令符合预期后，再执行：

```bash
DRY_RUN=0 \
JOB_NAME=<vcjob-name> \
MODE=static \
bash run_vcctl_healthcheck.sh
```

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

`MODE=all` 会依次执行三个独立阶段，每个阶段写入同一个 `RUN_ID` 下的独立目录：

```text
results/vcctl/<RUN_ID>/static/
results/vcctl/<RUN_ID>/single_node/
results/vcctl/<RUN_ID>/multi_node/
```

`pre-clean` 不是独立阶段；它会在每个阶段执行前清理残留的健康检查进程，并记录在该阶段的 `results.jsonl` 中。

### 4.4 通信路径探测

用于确认通信库实际可见的 HCA / rail、端口状态、IB counter 是否在 All-Reduce 前后增长，并采集 `MCCL_DEBUG` / `NCCL_DEBUG` 日志。

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

DRY_RUN=0 bash run_vcctl_comm_probe.sh
```

默认会在当前 vcjob 的每个 pod 上执行：

- 采集 `MCCL` / `NCCL` / `MACA` / `MASTER_*` / `RANK` / `WORLD_SIZE` 环境变量；
- 采集 `/sys/class/infiniband` 设备、端口、rate、GID、netdev 映射；
- 采集 `rdma link`、`rdma dev`、`ibv_devinfo`、`ip -br addr/link`；
- 在 All-Reduce 前后采集 IB counter，并生成 `ib_counters_delta.tsv`；
- 带 `MCCL_DEBUG=INFO`、`NCCL_DEBUG=INFO` 执行一次小规模 `run-bandwidth`。

输出默认写入：

```text
/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl_comm_probe/<RUN_ID>/
```

主要看每个 pod 的：

```text
pod_results/<pod>/multi-node/summary.md
pod_results/<pod>/multi-node/ib_sysfs_before.txt
pod_results/<pod>/multi-node/ib_counters_delta.tsv
pod_results/<pod>/multi-node/torch_debug.stderr
pod_results/<pod>/multi-node/torch_bandwidth/bandwidth_gate.json
```

## 5. MODE 与 PROFILE

| 参数 | 含义 |
| --- | --- |
| `MODE=static` | 执行 pod 内静态环境探测，并横向比对各 pod 输出 |
| `MODE=single-node` | 每个 pod 内单机 8 卡测试 |
| `MODE=multi-node` | 当前 vcjob 的多节点测试 |
| `MODE=all` | 依次执行 `static`、`single_node`、`multi_node` 三个独立结果目录 |
| `PROFILE=smoke` | 最短连通性测试，`nproc-per-node=1`，只做简单 `all_reduce` |
| `PROFILE=quick` | 快速正式测试，默认每节点 8 rank，覆盖计算、显存拷贝和通信算子 |
| `PROFILE=bandwidth` | All-Reduce 大包带宽 gate，默认 1G/4G/8G/16G、100 轮计时 |
| `PROFILE=collective-bandwidth` | 多 collective 大包带宽基线，覆盖 All-Reduce、Reduce-Scatter、All-Gather、All-to-All、EP=8 All-to-Allv |

## 6. 静态检查 compact 输出

`MODE=static` 默认使用 stdout frame 聚合输出。每个 pod 会先在 pod 本地 `/tmp` 下生成 raw 和 compact 临时文件，然后通过 stdout 返回一个 `__HC_STATIC_RESULT_JSON__` frame。driver 直接解析 frame 并生成 run 级结果。static 执行成功的 pod 不在共享存储保留 stdout / stderr；执行失败、超时或 frame 解析异常的 pod 会保留 stdout / stderr 到共享结果目录。

默认保留：

```text
results/vcctl/<run_id>/static/
  static_facts.jsonl       # 每行一个 pod / node 的 compact facts
  static_checks.jsonl      # 每行一个静态检查项结果
  static_failed_pods.jsonl # 超时、exec 失败、frame 解析失败的 pod
  static_compare.json
  static_compare.md
  static_outliers.jsonl
  summary.json
  summary.md
```

默认不长期保留：

```text
results/vcctl/<run_id>/static/pod_results/<pod>/static/
results/vcctl/<run_id>/static/logs/<pod>.static.stdout   # PASS pod 默认不保留
results/vcctl/<run_id>/static/logs/<pod>.static.stderr   # PASS pod 默认不保留
```

raw 裸日志默认保留在 pod 本地 `/tmp`，直到健康检测 job 被删除：

```text
/tmp/pretrain_healthcheck_<run_id>_<pod_name>_<pid>/static/
```

如需把 `mx-smi`、`ibv_devinfo` 等原始命令日志复制到共享存储：

```bash
STATIC_OUTPUT_MODE=raw STATIC_COPY_RAW_OUTPUT=1 STATIC_KEEP_POD_FILES=1 DRY_RUN=0 MODE=static bash run_vcctl_healthcheck.sh
```

## 7. quick 覆盖算子

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

## 8. All-Reduce 带宽 gate

`PROFILE=bandwidth` 用于对齐训练前集合通信带宽验收。该 profile 只执行 `all_reduce`，默认使用较大的 message size 连跑多轮，并按每轮 BusBW 的次低值和平均值做 gate。

默认配置：

| 项 | 默认值 |
| --- | --- |
| op | `all_reduce` |
| message size | `1G,4G,8G,16G` |
| warmup | `5` |
| iters | `100` |
| dtype | `bf16` |
| min gate | `270 GB/s` |
| avg gate | `290 GB/s` |

说明：`270/290 GB/s` 是脚本默认 gate 示例，不是当前两节点 MetaX C550 + MCCL 的已标定阈值。当前两节点 8GB All-Reduce 基线约为：

```text
avg_busbw:           93-96 GB/s
second_lowest_busbw: 88-92 GB/s
```

当前 MetaX 两节点环境建议先用 8GB 零阈值跑基线：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=bandwidth \
BANDWIDTH_MESSAGE_SIZES=8G \
BANDWIDTH_WARMUP=5 \
BANDWIDTH_ITERS=100 \
BANDWIDTH_MIN_BUSBW=0 \
BANDWIDTH_AVG_BUSBW=0 \
bash run_vcctl_healthcheck.sh
```

如果第一次只想验证链路和脚本是否能跑通，可以先降低测试量，并临时关闭阈值：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=bandwidth \
BANDWIDTH_MESSAGE_SIZES=1G \
BANDWIDTH_WARMUP=2 \
BANDWIDTH_ITERS=10 \
BANDWIDTH_MIN_BUSBW=0 \
BANDWIDTH_AVG_BUSBW=0 \
bash run_vcctl_healthcheck.sh
```

如果需要启用 gate，应先按当前硬件、通信库和节点规模标定阈值。例如当前两节点 MetaX C550 + MCCL 可先用较保守的 8GB gate：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=bandwidth \
BANDWIDTH_MESSAGE_SIZES=8G \
BANDWIDTH_WARMUP=5 \
BANDWIDTH_ITERS=100 \
BANDWIDTH_MIN_BUSBW=88 \
BANDWIDTH_AVG_BUSBW=92 \
bash run_vcctl_healthcheck.sh
```

判定逻辑：

```text
second_lowest_busbw > BANDWIDTH_MIN_BUSBW
avg_busbw           > BANDWIDTH_AVG_BUSBW
```

这里的 `second_lowest_busbw` 是 100 轮 BusBW 排序后的次低值，用来避免单轮极端抖动直接决定失败；`avg_busbw` 是 100 轮平均值，用来判断整体带宽水平。

当前不建议把 H200/NCCL 风格的 `270/290 GB/s` 阈值直接套用到 MetaX C550 + MCCL + 2 节点 16 rank 场景；该阈值已在 `results/vcctl/20260630_143416` 中验证为不匹配当前基线。

该 profile 输出：

| 文件 | 含义 |
| --- | --- |
| `bandwidth_round_detail.jsonl` | rank 级每轮 latency / algbw / busbw |
| `bandwidth_round_summary.jsonl` | group 级每轮 latency / algbw / busbw |
| `bandwidth_summary.jsonl` | 每个 message size 的次低值、平均值、min/max 和 gate 结果 |
| `bandwidth_gate.json` | 本次 bandwidth gate 的机器可读总结果 |
| `bandwidth_report.md` | 本次 bandwidth gate 的 Markdown 报告 |

## 9. 多 collective 带宽基线

`PROFILE=collective-bandwidth` 用于在 quick 功能测试之后，对关键 collective 做大包性能基线。当前筛查异常节点时，推荐把它作为“同批次多组横向对比”的采集项，而不是默认启用固定绝对阈值。

典型使用方式：

```text
同一轮测试
同一组参数
多个分组并行运行
比较各组 op / pattern / message size 下的 BusBW、latency、correctness
明显偏离同批其他分组的 group 标记为可疑
再通过换组、重配对或二分排查定位异常节点
```

它默认覆盖：

```text
all_reduce
reduce_scatter
all_gather
all_to_all
all_to_allv
```

其中 `all_to_allv` 按 `COLLECTIVE_BANDWIDTH_EP_SIZE=8` 创建 EP group，并使用 MoE payload patterns：

```text
uniform
skewed
hot_expert
random
empty_expert
```

默认配置：

| 项 | 默认值 |
| --- | --- |
| ops | `all_reduce,reduce_scatter,all_gather,all_to_all,all_to_allv` |
| message size | `1G` |
| warmup | `5` |
| iters | `30` |
| EP size | `8` |
| min gate | `0 GB/s`，表示默认不启用固定绝对阈值 |
| avg gate | `0 GB/s`，表示默认不启用固定绝对阈值 |

说明：该 profile 默认使用 `1G`，不是 `8G`。对 `all_gather`，这里把 message size 解释为最终 gathered payload 的总大小，每 rank 输入约为 `message_size / collective_group_size`，避免规模扩大后输出按 group size 继续放大导致 OOM。

`COLLECTIVE_BANDWIDTH_MIN_BUSBW` / `COLLECTIVE_BANDWIDTH_AVG_BUSBW` 保留为可选 absolute gate。当前千卡排查场景下，默认保持 `0/0`，由外层汇总多个 group 的 `collective_bandwidth_summary.jsonl` 后做横向对比。

两节点冒烟建议：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=collective-bandwidth \
COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=1M \
COLLECTIVE_BANDWIDTH_WARMUP=1 \
COLLECTIVE_BANDWIDTH_ITERS=1 \
bash run_vcctl_healthcheck.sh
```

两节点基线建议：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=collective-bandwidth \
COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=1G \
COLLECTIVE_BANDWIDTH_WARMUP=5 \
COLLECTIVE_BANDWIDTH_ITERS=30 \
COLLECTIVE_BANDWIDTH_MIN_BUSBW=0 \
COLLECTIVE_BANDWIDTH_AVG_BUSBW=0 \
bash run_vcctl_healthcheck.sh
```

如果只想测 MoE / EP 相关通信：

```bash
DRY_RUN=0 \
MODE=multi-node \
PROFILE=collective-bandwidth \
COLLECTIVE_BANDWIDTH_OPS=all_to_all,all_to_allv \
COLLECTIVE_BANDWIDTH_EP_SIZE=8 \
COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=1G \
bash run_vcctl_healthcheck.sh
```

该 profile 输出：

| 文件 | 含义 |
| --- | --- |
| `collective_bandwidth_round_detail.jsonl` | rank 级每轮 latency / algbw / busbw |
| `collective_bandwidth_round_summary.jsonl` | group 级每轮 latency / algbw / busbw |
| `collective_bandwidth_summary.jsonl` | op / message size / payload pattern 级汇总，包含本次 absolute gate 配置和实测值 |
| `collective_bandwidth_gate.json` | 本次 collective bandwidth absolute gate 的机器可读总结果；默认 `0/0` 时主要作为采集结果 |
| `collective_bandwidth_report.md` | 本次 collective bandwidth 的 Markdown 报告 |

## 10. 故障注入测试

故障注入用于验证健康检查工具自身是否能正确识别异常，并将异常传播到外层 `summary.md`。

### 10.1 NaN 精度异常

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

### 10.2 显式 corrupt 数据异常

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

### 10.3 hang / timeout

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

### 10.4 后端初始化失败

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

## 11. 单节点测试

### 11.1 MetaX C550 单节点 8 卡

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

### 11.2 NVIDIA H200 单节点 6 卡

NVIDIA 脚本保留用于单节点开发验证：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck

bash scripts/nvidia/dry_run_single_node_6h200.sh
bash scripts/nvidia/run_single_node_6h200.sh
```

说明：当前 README 主流程以 MetaX + vcctl 为准；NVIDIA 脚本没有参与当前两节点 vcctl 验证基线。

## 12. 静态环境探测

### 12.1 MetaX pod 能力探测

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

### 12.2 dmesg 策略

当前运维提供的 pod 不支持读取 `dmesg`，因此 probe 脚本不在 pod 内执行 `dmesg`。summary 中记录为：

```text
logs | dmesg | SKIP | pod environment does not support dmesg; kernel log screening is owned by ops
```

宿主机内核日志、PCIe AER、MCE、soft lockup、XID / NPU 历史错误筛查由运维侧提供结论。

### 12.3 静态结果横向比对

通过 `run_vcctl_healthcheck.sh` 执行 `MODE=static` 或 `MODE=all` 时，脚本会先并行到每个 pod 内运行 static probe，然后在 driver 侧读取每个 pod 的结果并做横向比对。

比对结果默认输出：

```text
results/vcctl/<run_id>/static/static_compare.json
results/vcctl/<run_id>/static/static_compare.md
results/vcctl/<run_id>/static/static_outliers.jsonl
```

static 比对采用每台机器独立采集后的多数投票基线，不使用二分定位。二分更适合集合通信这种“只知道某个组异常”的测试；static 结果本身已经带有 pod / node 归属，可以直接定位异常节点。

## 13. 输出目录与结果文件

vcctl 编排结果默认写入到 `RUN_ID` 下的阶段目录：

```text
${RESULT_ROOT}/${RUN_ID}/${STAGE}/
```

MetaX 默认：

```text
/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl/<RUN_ID>/<STAGE>/
```

阶段目录映射：

```text
MODE=static      -> static/
MODE=single-node -> single_node/
MODE=multi-node  -> multi_node/
MODE=all         -> static/ + single_node/ + multi_node/
```

每个阶段目录的主要文件：

| 文件 | 含义 |
| --- | --- |
| `summary.md` | 外层 vcctl Markdown 汇总；static 比对通过时包含一个代表性节点软硬件摘要 |
| `summary.json` | 外层 vcctl 机器可读汇总 |
| `results.jsonl` | 每个 pod / mode 的执行状态 |
| `commands.env` | 本次实际命令和关键环境变量 |
| `pods.jsonl` | 解析后的 pod 元信息 |
| `static_facts.jsonl` | static compact 聚合事实，每行一个 pod / node |
| `static_checks.jsonl` | static 检查项聚合结果，每行一个检查项 |
| `static_failed_pods.jsonl` | static 超时、exec 失败、frame 解析失败的 pod |
| `static_compare.json` | static 横向比对机器可读报告 |
| `static_compare.md` | static 横向比对 Markdown 报告 |
| `static_outliers.jsonl` | static 异常节点 / 异常字段明细 |
| `logs/<pod>.<mode>.stdout` | 每个 pod / mode 的 stdout；static PASS pod 默认不保留 |
| `logs/<pod>.<mode>.stderr` | 每个 pod / mode 的 stderr；static PASS pod 默认不保留 |
| `pod_results/<pod>/<mode>/` | 注入给检测程序的 pod 结果目录；static 默认不生成共享存储 pod 级目录 |

动态测试的 pod 结果目录中常见文件：

| 文件 | 含义 |
| --- | --- |
| `report.md` | 分析器生成的 Markdown 报告 |
| `group_summary.jsonl` | group 级 op / message size / payload pattern 汇总 |
| `rank_detail.jsonl` | rank 级明细，用于定位慢 rank、NaN、Inf、错误 rank |
| `ping_summary.json` | smoke 模式的最小 all_reduce 连通性结果 |
| `bandwidth_round_detail.jsonl` | bandwidth 模式的 rank 级每轮带宽明细 |
| `bandwidth_round_summary.jsonl` | bandwidth 模式的 group 级每轮带宽汇总 |
| `bandwidth_summary.jsonl` | bandwidth 模式的 message size 级 gate 汇总 |
| `bandwidth_gate.json` | bandwidth 模式的总 gate 结果 |
| `bandwidth_report.md` | bandwidth 模式的 Markdown 报告 |
| `collective_bandwidth_round_detail.jsonl` | collective-bandwidth 模式的 rank 级每轮带宽明细 |
| `collective_bandwidth_round_summary.jsonl` | collective-bandwidth 模式的 group 级每轮带宽汇总 |
| `collective_bandwidth_summary.jsonl` | collective-bandwidth 模式的 op / pattern 级 gate 汇总 |
| `collective_bandwidth_gate.json` | collective-bandwidth 模式的总 gate 结果 |
| `collective_bandwidth_report.md` | collective-bandwidth 模式的 Markdown 报告 |
| `ib_counters_delta.tsv` | comm probe 模式的 IB counter 增量；当前 pod 内标准 counter 未暴露时可能只有表头 |
| `torch_debug.stderr` | comm probe 模式的 torch / MCCL debug stderr |
| `torch_debug.stdout` | comm probe 模式的 MCCL debug stdout，可查看 `NET/IB`、`MCCL_IB_HCA`、`xscale_*` |

## 14. 常用环境变量

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `JOB_NAME` | `muxi-2node` | vcjob 名称，用于 `vcctl pod get --job` |
| `NAMESPACE` | `default` | namespace |
| `MODE` | `all` | `static`、`single-node`、`multi-node`、`all` |
| `DEVICE_TYPE` | `metax` | 结果元信息，例如 `gpu`、`npu`、`metax` |
| `PROJECT_REMOTE_DIR` | `/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck` | pod 内项目路径 |
| `PROFILE` | `quick` | `quick`、`smoke`、`bandwidth` 或 `collective-bandwidth` |
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
| `STATIC_COMPARE` | `1` | `MODE=static/all` 后是否横向比对 static 输出 |
| `STATIC_COMPARE_WORKERS` | `0` | static 比对解析并发数，`0` 表示自动 |
| `STATIC_COMPARE_STRICT` | `1` | static 比对发现异常时是否影响 overall status |
| `STATIC_EXPECTED_GPUS` | `GPUS_PER_NODE` | static rule gate 期望 GPU 数；`0` 表示关闭 |
| `STATIC_EXPECTED_XSCALE_PORTS` | `0` | static rule gate 期望 xscale/HCA 端口数；`0` 表示关闭 |
| `STATIC_OUTPUT_MODE` | `compact` | static probe 输出模式；`raw` 可配合 `STATIC_COPY_RAW_OUTPUT=1` 复制原始日志到共享存储 |
| `STATIC_TMP_ROOT` | `/tmp` | pod 内 static raw 临时文件根目录 |
| `STATIC_KEEP_LOCAL_TMP` | `1` | 是否保留 pod 本地 `/tmp/pretrain_healthcheck_*` 临时目录 |
| `STATIC_COPY_RAW_OUTPUT` | `0` | 是否把 raw 临时文件复制到共享结果目录 |
| `STATIC_STDOUT_MAX_BYTES` | `1048576` | 单 pod static stdout frame 最大字节数 |
| `STATIC_EXEC_TIMEOUT_SECONDS` | `120` | 单 pod static probe 超时时间 |
| `STATIC_DRIVER_TMP_ROOT` | `/tmp` | 开发机侧 static stdout/stderr 临时目录 |
| `STATIC_KEEP_POD_FILES` | `0` | static 聚合后是否保留 `pod_results/<pod>/static` |
| `STATIC_KEEP_EXEC_LOGS` | `0` | static 是否保留 PASS pod 的 stdout/stderr |
| `BANDWIDTH_MESSAGE_SIZES` | `1G,4G,8G,16G` | `PROFILE=bandwidth` 的 All-Reduce payload 大小列表 |
| `BANDWIDTH_WARMUP` | `5` | `PROFILE=bandwidth` 的 warmup 轮数 |
| `BANDWIDTH_ITERS` | `100` | `PROFILE=bandwidth` 的计时轮数 |
| `BANDWIDTH_MIN_BUSBW` | `270` | `PROFILE=bandwidth` 的次低 BusBW gate，单位 GB/s |
| `BANDWIDTH_AVG_BUSBW` | `290` | `PROFILE=bandwidth` 的平均 BusBW gate，单位 GB/s |
| `COLLECTIVE_BANDWIDTH_OPS` | `all_reduce,reduce_scatter,all_gather,broadcast,all_to_all,all_to_allv` | `PROFILE=collective-bandwidth` / `PROFILE=dynamic-suite` 的 op 列表 |
| `COLLECTIVE_BANDWIDTH_MESSAGE_SIZES` | `1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G` | `PROFILE=collective-bandwidth` / `PROFILE=dynamic-suite` 的 payload 大小列表 |
| `COLLECTIVE_BANDWIDTH_WARMUP` | `5` | `PROFILE=collective-bandwidth` 的 warmup 轮数 |
| `COLLECTIVE_BANDWIDTH_ITERS` | `30` | `PROFILE=collective-bandwidth` 的计时轮数 |
| `COLLECTIVE_BANDWIDTH_MIN_BUSBW` | `0` | `PROFILE=collective-bandwidth` 的次低 BusBW gate，单位 GB/s |
| `COLLECTIVE_BANDWIDTH_AVG_BUSBW` | `0` | `PROFILE=collective-bandwidth` 的平均 BusBW gate，单位 GB/s |
| `COLLECTIVE_BANDWIDTH_EP_SIZE` | `8` | `PROFILE=collective-bandwidth` 中 `all_to_allv` 的 EP group size |
| `SEED` | `20260623` | 随机种子 |
| `RESULT_ROOT` | `/afs-a3-weight-share/zhangcaixian/scale_up10000/pretrain_healthcheck/results/vcctl` | 共享结果根目录 |
| `RUN_ID` | 当前时间戳 | 本次运行 ID |
| `EXEC_TIMEOUT_SECONDS` | `180` | 每个 pod exec 的超时时间；完整 collective 矩阵默认自动提升到 `1800` |
| `MAX_PARALLEL` | `0` | 最大并发 pod 数，`0` 表示全部并发 |
| `CONTAINER_NAME` | 空 | 强制指定 container；为空时自动选择 |
| `DRY_RUN` | `1` | 执行前预演 / 前置检查模式；`1` 表示只解析 pod、打印命令，不真正执行 pod 内检测 |
| `POD_JSON_FILE` | 空 | 使用本地 pod JSON 文件做解析测试，不调用 `vcctl` |

故障注入变量：

| 环境变量 | 含义 |
| --- | --- |
| `FAULT_BACKEND=1` | 强制使用非法 backend，验证初始化失败采集 |
| `FAULT_SLEEP_RANK=<rank>` | 指定全局 rank 在测试中 sleep，模拟慢 rank / hang |
| `FAULT_SLEEP_SECONDS` | sleep 秒数，默认 `30` |
| `FAULT_NAN_RANK=<rank>` | 指定全局 rank 注入 NaN |
| `FAULT_CORRUPT_RANK=<rank>` | 指定全局 rank 修改 tensor，模拟结果污染 |

## 15. 已验证结果

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

已验证 bandwidth / 通信路径：

| 测试项 | 代表目录 | 结果 |
| --- | --- | --- |
| 1GB bandwidth 冒烟 | `results/vcctl/20260630_142910` | PASS，avg_busbw 约 89.05 GB/s |
| 270/290 gate 验证 | `results/vcctl/20260630_143416` | FAIL，说明该阈值不匹配当前 MetaX 两节点基线 |
| 1G/4G/8G bandwidth 基线 | `results/vcctl/20260630_144024` | PASS，8GB 最稳定，avg_busbw 约 95.21 GB/s |
| 通信路径探测 | `results/vcctl_comm_probe/comm_probe_20260630_145457` | PASS，MCCL 数据面走 `xscale_0..xscale_3 / RoCE / IB`，`eth0` 仅用于 bootstrap/OOB |
| 8GB bandwidth 复测 | `results/vcctl/20260630_150653` | PASS，avg_busbw 约 93.73 GB/s，second_lowest 约 87.78 GB/s |
| collective-bandwidth 冒烟 | `results/vcctl/20260630_173604` | PASS，1MB / 1 iter，覆盖 9 个 case |
| 1GB collective-bandwidth 基线 | `results/vcctl/20260630_174255` | PASS，9 个 case / 30 iter，`all_to_all` 有两轮低谷 |
| 1GB collective-bandwidth 复测 | `results/vcctl/20260630_174615` | PASS，9 个 case / 30 iter，`all_to_all` 低谷未复现 |

完整测试结果总结见：

```text
../design_docs/pretrain_healthcheck_metax_test_summary.md
```

## 16. 同硬件环境 clone YAML

训练 job 异常后，如果需要恢复同一批物理机器上的软件 / 硬件环境，可先在原 job 还存在时导出原始 job 配置、pod 到 node 的映射，并生成固定节点的 clone YAML。

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

JOB_NAME=muxi-2node1 \
bash export_same_node_clone_yaml.sh
```

默认输出：

```text
results/job_clone/<job_name>_<run_id>/
  <job_name>.yaml
  <job_name>.json
  node_map.txt
  <job_name>_clone.yaml
```

`<job_name>_clone.yaml` 会清理原 job 的运行时字段，并通过 `kubernetes.io/hostname` nodeAffinity 绑定到原 pod 所在的物理节点。clone job 保持原容器启动命令不变，即 `sshd + sleep inf`，用于先恢复环境，再通过 `vcctl exec` 运行健康检测脚本。

精确绑定规则：

- `master replicas=1` 保持原 task 名 `master`，绑定到原 `master-0` 所在节点。
- `worker replicas=N` 会展开成 N 个单副本 task，例如 `worker-0`、`worker-1`、`worker-2`。
- 展开后的每个 worker task 分别绑定到原同名 pod 所在节点，例如 task `worker-1` 绑定原 `worker-1` 的 `nodeName`。
- `spec.plugins.pytorch` 中的 `--worker=worker` 会同步展开为多个 `--worker=worker-0`、`--worker=worker-1`。

恢复环境：

```bash
vcctl job run -f results/job_clone/<job_name>_<run_id>/<job_name>_clone.yaml -n default
```

注意事项：

- 默认使用 nodeAffinity，不直接写 `nodeName`，以匹配当前平台已有调度方式。
- 当前只对 PyTorch 插件中的 worker 多副本做自动展开；其他多副本 task 会拒绝生成，避免误改语义。
- 第一次在新平台使用时，建议先只生成 YAML 并人工检查，不直接执行 `vcctl job run`。

## 17. 单节点完整可运行性检测

如果只关注单节点连通性、可运行性和节点内多卡通信，可以一条命令执行：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

JOB_NAME=<vcjob-name> \
DRY_RUN=0 \
bash run_single_node_full_healthcheck_with_sync.sh
```

该脚本会使用同一个 `RUN_ID` 串行执行：

```text
static
dynamic_suite
```

`dynamic_suite` 只启动一次 `torchrun`，在同一个 distributed process group 内依次执行 `smoke`、`quick`、`bandwidth` 和 `collective-bandwidth`，避免大规模节点上反复初始化通信组。`collective-bandwidth` 默认覆盖 `1K` 到 `2G` 的 22 个包大小，以及 `all_reduce`、`reduce_scatter`、`all_gather`、`broadcast`、`all_to_all`、`all_to_allv` 六类算子。`bandwidth` 和 `collective-bandwidth` 默认关闭固定阈值 gate，主要用于同批节点横向比较。

结果目录：

```text
results/vcctl/<RUN_ID>/
  static/
  dynamic_suite/
```

动态阶段的每个 pod 会把详细原始结果写到 pod 本地：

```text
/tmp/pretrain_healthcheck_<RUN_ID>_<pod_name>_<pid>/dynamic_suite/
  smoke/
  quick/
  bandwidth/
  collective_bandwidth/
```

共享 `results/` 目录默认只保留 compact 聚合和比对结果：

```text
commands.env
pods.jsonl
results.jsonl
summary.json
summary.md
dynamic_facts.jsonl
dynamic_failed_pods.jsonl
dynamic_compare.json
dynamic_compare.md
dynamic_outliers.jsonl
```

成功 pod 的共享 stdout/stderr 和 `pod_results/` 默认不保留；失败、超时、frame 解析失败时才保留 `logs/` 用于排查。

## 18. 多节点分组 batch 检测

静态检测和单节点 dynamic-suite 通过后，可以在一个已经申请好的大 job 内执行多节点分组检测。该入口只负责多节点 group，不会重新执行 static 或 single-node。

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck

JOB_NAME=muxi-1024node \
TARGET_SCALE=128 \
DRY_RUN=0 \
PRE_CLEAN=1 \
DYNAMIC_COMPARE=1 \
GROUP_TIMEOUT_SECONDS=180 \
bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh
```

脚本会读取当前 job 的 pod 列表，按固定 seed 生成分组，并为每个 group 生成临时 pod JSON，复用：

```text
MODE=multi-node
PROFILE=dynamic-suite
POD_JSON_FILE=<group_pods.json>
```

默认 phase 根据节点数和 `TARGET_SCALE` 自动选择：

```text
pairwise
ep8
scale64
scale128
scale256
```

也可以显式指定：

```bash
PHASES=pairwise,ep8,scale64
```

共享结果目录：

```text
results/vcctl/<BATCH_RUN_ID>/
  batch_results.sqlite
  batch_summary.md
  batch_summary.json
  pass_nodes.txt
  suspect_nodes.txt
  fail_nodes.txt
  node_map.txt
  failed_groups/
```

正常 group 的共享存储明细默认会被清理，只保留 SQLite 和 batch 汇总；失败 group 会在 `failed_groups/<group_id>/summary.md` 保留摘要。pod 内原始动态测试明细仍保留在对应 pod 的 `/tmp/pretrain_healthcheck_*` 目录中。

同一轮内的 group 默认并发执行：

```text
PHASE_GROUP_CONCURRENCY=0
```

`0` 表示当前 round 内所有互不重叠的 group 一起启动。例如 1024 节点 `pairwise_r1` 会同时启动 512 个二节点 group；`pairwise_r2` 会在 `pairwise_r1` 完成后再启动。不同 round 之间仍然串行，避免同一节点同时参与多个 group。若开发机 CPU、进程数或 API server 压力需要限流，可以设置：

```bash
PHASE_GROUP_CONCURRENCY=64
```

断点续跑：

```bash
JOB_NAME=muxi-1024node \
BATCH_RUN_ID=<existing_batch_id> \
bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh --resume
```

本地 dry-run 可以使用保存的 pod JSON：

```bash
JOB_NAME=muxi-2node1 \
POD_JSON_FILE=/tmp/pods.json \
PHASES=pairwise \
DRY_RUN=1 \
bash scripts/metax/run_vcctl_multi_node_batch_healthcheck.sh
```

注意：该 batch runner 的 group 临时 pod JSON 写在开发机 `/tmp/pretrain_healthcheck_batch_<BATCH_RUN_ID>/`，不写共享存储。

## 19. 当前限制

- 当前 vcctl 多节点验证只在两节点 MetaX C550 环境完成闭环。
- 更大规模的 8 / 64 / 128 节点测试需要结合实际资源申请和分组策略继续验证。
- pod 内不执行 `dmesg`，宿主机内核日志筛查由运维侧提供。
- static probe 中部分系统命令可能在 pod 内缺失，例如 `rdma`、`ip`、`numactl`；这类结果用于能力探测，不等价于训练不可用。
- pod 内标准 `/sys/class/infiniband/.../counters` 当前未暴露有效 counter，通信路径探测只能从 MCCL debug 确认 `xscale_0..xscale_3` 被识别和启用，不能在 pod 内直接证明每条 rail 的实际流量占比。
- 当前两节点 MetaX C550 + MCCL 的 8GB All-Reduce 基线约为 `avg_busbw 93-96 GB/s`、`second_lowest_busbw 88-92 GB/s`；该数据只作为两节点参考基线。
- `PROFILE=collective-bandwidth` / `PROFILE=dynamic-suite` 的 collective-bandwidth 阶段默认使用 `1K,2K,4K,8K,16K,32K,64K,128K,256K,512K,1M,2M,4M,8M,16M,32M,64M,128M,256M,512M,1G,2G`；其中 `all_gather` 的 message size 表示 gathered 后的总 payload，`all_to_allv` 按 MoE payload pattern 生成非均匀 split，其余 op 表示本 rank 参与 collective 的输入 payload。当前异常节点筛查推荐以同批次多组横向对比为主，固定 absolute gate 默认关闭。
- 当前 NPU / HCCL 完整执行路径尚未作为主流程验证。
- 同硬件环境 clone YAML 支持 `master replicas=1` 加 PyTorch `worker replicas>=1` 的精确恢复；其他多副本 task 暂不自动展开。

## 20. 建议使用顺序

训练前建议按以下顺序执行：

1. `PROFILE=smoke MODE=multi-node`：快速确认 vcctl、torchrun、多 pod 连通性。
2. `MODE=all PROFILE=quick MESSAGE_SIZES=1M WARMUP=1 ITERS=1`：完整快速验收。
3. `MODE=multi-node PROFILE=bandwidth BANDWIDTH_MESSAGE_SIZES=8G BANDWIDTH_MIN_BUSBW=0 BANDWIDTH_AVG_BUSBW=0`：采集当前 8GB All-Reduce 基线。
4. `MODE=multi-node PROFILE=collective-bandwidth COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=1G COLLECTIVE_BANDWIDTH_MIN_BUSBW=0 COLLECTIVE_BANDWIDTH_AVG_BUSBW=0`：采集关键 collective 大包基线，包含 EP=8 all_to_allv。
5. `DRY_RUN=0 bash run_vcctl_comm_probe.sh`：确认通信数据面是否走 `xscale/RoCE/IB`。
6. 多组筛查时，汇总各组 `bandwidth_summary.jsonl` / `collective_bandwidth_summary.jsonl`，对同一 op、message size、payload pattern 做横向对比，明显偏离同批其他组的 group 标记为可疑。
7. 若确实需要固定 absolute gate，再基于当前硬件和规模标定 `BANDWIDTH_*` / `COLLECTIVE_BANDWIDTH_*` 阈值；默认不依赖固定阈值筛节点。
8. 若工具变更后需要自测，依次运行 NaN、corrupt、timeout、backend failure 故障注入。

常用命令：

```bash
cd /mnt/hgfs/nfs_share/ailab/scale_up10000/pretrain_healthcheck/scripts/metax

DRY_RUN=0 MODE=multi-node PROFILE=smoke bash run_vcctl_healthcheck.sh

DRY_RUN=0 MODE=all PROFILE=quick MESSAGE_SIZES=1M WARMUP=1 ITERS=1 bash run_vcctl_healthcheck.sh

DRY_RUN=0 MODE=multi-node PROFILE=bandwidth BANDWIDTH_MESSAGE_SIZES=8G BANDWIDTH_MIN_BUSBW=0 BANDWIDTH_AVG_BUSBW=0 bash run_vcctl_healthcheck.sh

DRY_RUN=0 MODE=multi-node PROFILE=collective-bandwidth COLLECTIVE_BANDWIDTH_MESSAGE_SIZES=1G COLLECTIVE_BANDWIDTH_MIN_BUSBW=0 COLLECTIVE_BANDWIDTH_AVG_BUSBW=0 bash run_vcctl_healthcheck.sh

DRY_RUN=0 bash run_vcctl_comm_probe.sh
```
