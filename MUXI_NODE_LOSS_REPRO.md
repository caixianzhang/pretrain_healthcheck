# 沐曦 96/128 节点失联复现说明

该入口用于复现已观察到的现象：高并发训练拓扑通信与 Pod、GPU、RDMA 健康状态下降存在时间相关性。它属于厂家专项诊断负载，不是常规训练准入测试。

## 安全边界

- 负载使用 TP=4、EP=32、ETP=1、PP=8、CP=1。
- TP 和 PP payload 为 32 MiB，EP payload 为 64 MiB。
- 数据类型为 bf16，执行 1 次 warmup 和 1 次正式迭代。
- 不执行 768/1024 ranks 全局 All-to-All。
- 不执行 `mx-smi --kill-all-process`。
- 清理操作只匹配包含当前 `REPRO_RUN_ID` 的进程。
- 正式执行必须显式设置 `CONFIRM_NODE_LOSS_REPRO=YES`。
- 开发机侧拓扑导出默认使用 `DRIVER_PYTHON=/opt/conda/bin/python3` 和 `MACA_HOME=/opt/maca`。

不要在正在执行训练任务或其他集合通信任务的 Job 上运行该复现程序。

## 排除已知异常节点

`EXCLUDED_NODES` 接受逗号、空格或换行分隔的 hostname，采用区分大小写的精确匹配：

```bash
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168
```

控制器按以下顺序处理节点：

1. 读取 Job 的完整 Pod 和 hostname 列表。
2. 校验每个请求排除的 hostname。
3. 移除匹配的节点。
4. 校验所有未排除 Pod 的 Running、Ready 和 Pod IP 状态。
5. 按 task 和 Pod 序号的稳定顺序选择前 96 或 128 个 Pod。

未知 hostname 会被视为配置错误。排除后可用节点少于目标规模时，程序返回退出码 `20`，不会静默降低测试规模。

只应排除已经由平台失联状态、静态硬件证据或厂家诊断确认的节点。不要把一个失败通信组的全部成员直接加入排除列表。

## 只执行预检

预检会生成准确的节点选择和训练拓扑计划，但不会启动集合通信，也不要求设置风险确认参数：

```bash
cd /afs-a3-241ceshi-shared/zhangcaixian/scale_up10000/pretrain_healthcheck

JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168 \
PREFLIGHT_ONLY=1 \
bash scripts/metax/run_vcctl_96node_loss_repro.sh
```

重点检查以下文件：

```text
results/vcctl/<RUN_ID>/
  excluded_nodes_requested.txt
  excluded_nodes_matched.tsv
  eligible_nodes.txt
  inputs/selected_nodes.txt
  standby_nodes.txt
  process_preflight.json
  topology_profile_summary.json
  reproduction_manifest.json
```

## 正式执行 96 节点复现

```bash
JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES=host-10-12-144-79,host-10-12-144-168 \
CONFIRM_NODE_LOSS_REPRO=YES \
bash scripts/metax/run_vcctl_96node_loss_repro.sh
```

## 正式执行 128 节点复现

排除异常节点后必须仍有 128 个 Running/Ready 的候选 Pod：

```bash
JOB_NAME=muxi-128node-4 \
EXCLUDED_NODES= \
CONFIRM_NODE_LOSS_REPRO=YES \
bash scripts/metax/run_vcctl_128node_loss_repro.sh
```

每轮正式测试依次执行：

1. 检查 Pod 成员、Ready 状态、Pod IP、每节点 8 张卡以及遗留进程。
2. 执行 120 秒空闲存活基线，每 5 秒采样一次 Job 状态。
3. 在选中节点上执行训练拓扑通信。
4. 发现存活状态下降后立即停止通信并进行定向清理。
5. 继续观察 120 秒，捕获延迟出现的 GPU 或 RDMA unhealthy 状态。
6. 提取错误特征并生成下一轮建议排除列表。

## 查看结果

```bash
bash scripts/metax/inspect_vcctl_node_loss_repro.sh \
  results/vcctl/<RUN_ID>
```

主要结果文件：

| 文件 | 含义 |
| --- | --- |
| `run_summary.md/json` | 最终状态、退出码、耗时和测试规模 |
| `lost_nodes.tsv` | 失联 Pod、hostname、是否属于通信组、原因和平台消息 |
| `suggested_excluded_nodes.txt` | 建议人工确认后用于下一轮的 hostname |
| `liveness_events.jsonl` | 按时间记录的 Ready、Pod IP、UID 和 Phase 变化 |
| `job_liveness_alert.json` | 首次检测到的存活状态变化 |
| `job_state_first_failure.json` | 首次失败处理时的完整 Job 快照 |
| `job_state_after_30s.json` | 失败 30 秒后的完整 Job 快照 |
| `job_state_after_120s.json` | 观察窗口结束后的完整 Job 快照 |
| `error_signatures.jsonl` | QP、ATU、illegal memory、NET/IB、admission 和 unhealthy 错误特征 |
| `evidence/error_excerpt.log` | 从控制器和结果日志中提取的可读错误片段 |
| `raw_output` | 指向开发机本地 `/tmp` 详细输出的软链接 |

程序不会自动应用 `suggested_excluded_nodes.txt`。再次执行前必须检查异常原因和平台状态，再将确认过的 hostname 合并到 `EXCLUDED_NODES`。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 通信完成且未出现节点失联，或预检通过 |
| 10 | 选中的通信节点健康或 Ready 状态下降，复现成功 |
| 11 | 未参与通信的备用 Job 节点健康或 Ready 状态下降 |
| 20 | 预检失败，或可用节点不足目标规模 |
| 30 | 通信失败，但平台节点状态没有下降 |
| 40 | 编排器内部错误 |

退出码 `30` 不能证明发生了节点失联。退出码 `10` 必须同时存在选中通信节点的平台存活状态变化证据。
