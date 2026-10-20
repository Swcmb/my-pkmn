# pkmn-kaggle 实际 steps/s 基准测试报告

- **任务来源**：`Prompt/pkmn-kaggle 实际 steps-s 基准测试与训练时间估算.md`
- **执行日期**：2026-10-17
- **执行主机**：Windows 原生 + conda 环境 `deeplearning` + NVIDIA GeForce RTX 5060 Laptop GPU
- **基准脚本**：`rl/train_selfplay.py`（自对弈 PPO 训练入口）
- **主结论**：稳定段实测 **307.67 env steps/s**（`avg` 累计列 302.2）。按此速率：10M env steps ≈ **9.0 小时**，100M ≈ **3.8 天**，1B ≈ **37.6 天**，11B ≈ **413.8 天**。

---

## 1. 结论速览

| 目标 env steps | 主口径 307.67 steps/s | 折算 |
|---:|---:|---:|
| 10M | 9.0 h | — |
| 100M | 3.8 d | 90.3 h |
| 1B | 37.6 d | 902.8 h |
| 11B | 413.8 d | 9931 h |

- 口径公式：`时长 = target_steps / (steps_per_sec × 3600)`，单位小时；天 = 小时 / 24。
- 主口径取 **it=10–30 稳定段 `sps` 均值 = 307.67**（21 次迭代，逐行见 §4.1），理由：直接反映纯训练迭代速度，避免启动/收尾段与全局累计平均的稀释。
- 对照：任务文档参考值 200 steps/s → 11B 需 636.6 天；本机实测比参考快约 1.54 倍。

---

## 2. 运行环境（含与任务建议的偏差标注）

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 原生（非 WSL2，见下方偏差说明） |
| Python | 3.12.14（`D:\ProgramFiles\deeplearning\python.exe`，conda env `deeplearning`） |
| PyTorch | 2.14.0+cu130，CUDA 13.0 runtime，`torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，sm_120（计算能力 12.0） |
| 驱动 | 581.29（WDDM 模式，`Bus-Id 00000000:01:00.0`） |
| 空闲基线 | GPU util 0%，显存占用 46 MiB |

**偏差说明（WSL2）**：任务文档建议 WSL2 路径；实际执行采用 **Windows 原生 conda 的 `deeplearning` 环境**。该环境已具备可用的 CUDA 13.0 torch 且 GPU 直接可见（无需 WSL CUDA 转发），训练循环与引擎（CABT）均在同一进程内，steps/s 口径不受虚拟化层影响。本报告全部数字均来自 Windows 原生路径。

**引擎建立方式（重要）**：PyPI wheel 自带 CABT 二进制，但 `kaggle_environments/__init__.py` 存在 Flask 强制导入，直接 import 会失败。解决方式为最小 shim：从 wheel 抽取 `envs/cabt/cg/*` 原字节 + 3 个空 `__init__.py`，置于 `PYTHONPATH=C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim`。此后 `battle_start(deck0, deck1, reverse_player=False)` 冒烟通过。

---

## 3. 基准配置与调用命令

### 3.1 完整调用命令（正式基准，exit 0）

```powershell
cd D:\pkmn-kaggle
$env:PYTHONPATH = "C:\Users\Swcmb\AppData\Local\Temp\opencode\bench-shim"
$env:PYTHONUNBUFFERED = "1"
D:\ProgramFiles\deeplearning\python.exe -m rl.train_selfplay `
  --decks sample --num-envs 8 --num-steps 64 --num-minibatches 16 `
  --total-timesteps 32768 --log-every 1 --out "runs\bench-rtx5060"
```

### 3.2 关键配置（训练启动时打印）

| 字段 | 值 | 说明 |
|---|---|---|
| arch / seed / device | transformer2 / 1 / cuda | 模型与设备 |
| decks | sample | 88 张样例卡组全集 |
| num_envs | 8 | 并行环境数 |
| num_steps | 64 | 每环境 rollout 长度 |
| batch_size | 512 | = num_envs × num_steps = 8 × 64 |
| num_minibatches | 16 | minibatch_size = 32（512/16） |
| update_epochs | 1 | PPO 更新轮数 |
| total_timesteps | 32768 | 本次基准总步数（= 64 次迭代 × 512） |
| log_every | 1 | 每次迭代打印一行 |
| native_encode | False | 见 §9.4 效力标注 |
| compile | False | 未启用 torch.compile |
| out | runs\bench-rtx5060 | 检查点输出目录 |
| 网络参数 | **[net] params=1,079,427** | 约 1.08M 参数（transformer2） |

迭代数学：`num_iters = total_timesteps / batch_size = 32768 / 512 = 64`；`sps = int(batch × world / max(collect+update, ε))`（当次迭代实时速率）；`avg = int((global_step - resume_step) / elapsed)`（全进程累计平均）。全程 `tch_wr=nan`（teacher 关闭）。

---

## 4. steps/s 实测结果

### 4.1 稳定段 it=10–30 全量逐迭代（n=21）

| it | sps | avg | it | sps | avg | it | sps | avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 326 | 299 | 17 | 318 | 304 | 24 | 301 | 303 |
| 11 | 323 | 301 | 18 | 280 | 303 | 25 | 270 | 301 |
| 12 | 323 | 303 | 19 | 343 | 305 | 26 | 268 | 300 |
| 13 | 262 | 299 | 20 | 280 | 303 | 27 | 340 | 301 |
| 14 | 345 | 302 | 21 | 273 | 302 | 28 | 349 | 303 |
| 15 | 327 | 304 | 22 | 339 | 303 | 29 | 278 | 302 |
| 16 | 299 | 303 | 23 | 294 | 303 | 30 | 323 | 302 |

**统计**：

| 指标 | sps 列 | avg 列 |
|---|---:|---:|
| n | 21 | 21 |
| 均值 | **307.67** | **302.19** |
| 最小 | 262（it=13） | 299（it=10/13） |
| 最大 | 349（it=28） | 305（it=19） |
| 标准差（样本） | 29.07 | 1.63 |
| 求和 | 6461 | 6346 |

说明：`sps` 为逐迭代实时速率（波动区间 262–349），`avg` 为全进程累计平均（收敛在 299–305 窄带内，末值 305）。

### 4.2 末段 it=55–64 逐迭代

| it | sps | avg | it | sps | avg |
|---:|---:|---:|---:|---:|---:|
| 55 | 220 | 304 | 60 | 332 | 303 |
| 56 | 306 | 304 | 61 | 321 | 304 |
| 57 | 268 | 304 | 62 | 326 | 304 |
| 58 | 277 | 303 | 63 | 300 | 304 |
| 59 | 297 | 303 | 64 | 379 | 305 |

it=55 的 220 与 it=64 的 379 属收尾段波动（it=64 为无后继负载的末迭代）；`avg` 列终值 305。

### 4.3 交付物：训练日志最后 10 行（逐字）

以下为 `bench-gpu.log` 尾部**最后 10 行训练日志**（完整字段，含 pg/v_ce/ent/clip/kl）：

```text
step=28160 it=55/64 sps=220 avg=304 tch_wr=nan pg=0.001 v_ce=0.000 ent=0.679 clip=0.000 kl=0.0000
step=28672 it=56/64 sps=306 avg=304 tch_wr=nan pg=-0.006 v_ce=0.000 ent=0.616 clip=0.000 kl=0.0003
step=29184 it=57/64 sps=268 avg=304 tch_wr=nan pg=0.004 v_ce=0.000 ent=0.610 clip=0.021 kl=0.0014
step=29696 it=58/64 sps=277 avg=303 tch_wr=nan pg=-0.001 v_ce=0.000 ent=0.673 clip=0.000 kl=0.0001
step=30208 it=59/64 sps=297 avg=303 tch_wr=nan pg=-0.023 v_ce=0.000 ent=0.752 clip=0.000 kl=0.0003
step=30720 it=60/64 sps=332 avg=303 tch_wr=nan pg=-0.000 v_ce=0.000 ent=0.834 clip=0.000 kl=0.0002
step=31232 it=61/64 sps=321 avg=304 tch_wr=nan pg=-0.010 v_ce=0.000 ent=0.814 clip=0.002 kl=0.0008
step=31744 it=62/64 sps=326 avg=304 tch_wr=nan pg=-0.004 v_ce=0.000 ent=0.632 clip=0.010 kl=0.0007
step=32256 it=63/64 sps=300 avg=304 tch_wr=nan pg=0.009 v_ce=0.000 ent=0.821 clip=0.002 kl=0.0002
step=32768 it=64/64 sps=379 avg=305 tch_wr=nan pg=-0.026 v_ce=0.000 ent=0.751 clip=0.000 kl=0.0003
```

日志文件真正结尾的 2 行（非训练日志行，收尾动作）：

```text
[ckpt] saved runs\bench-rtx5060\ckpt_32768.pt
done.
```

---

## 5. 训练中 GPU 采样（交付物）

### 5.1 nvidia-smi 快照（逐字，全部 52 行）

来源：第二次同参长跑（`bench-smi2`：同参数 32768 步、`--log-every 4`，exit 0，末行 `step=32768 it=64/64 sps=324 avg=314`），在训练进行中 t≈40s / 60s 各抓一张。文件 `nvidia-smi-midrun.txt` 全文如下：

```text
==== snapshot 1 @ 20:39:17 ====
Sat Oct 17 20:39:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.29                 Driver Version: 581.29         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   79C    P4             37W /  100W |     643MiB /   8151MiB |     12%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            7928    C+G   ...ng\Sysdiag\bin\HipsDaemon.exe      N/A      |
|    0   N/A  N/A           15940      C   ...Files\deeplearning\python.exe      N/A      |
+-----------------------------------------------------------------------------------------+
---- compute apps ----
pid, process_name, used_gpu_memory [MiB]
7928, E:\Program Files\Huorong\Sysdiag\bin\HipsDaemon.exe, [N/A]
15940, D:\ProgramFiles\deeplearning\python.exe, [N/A]
==== snapshot 2 @ 20:39:37 ====
Sat Oct 17 20:39:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.29                 Driver Version: 581.29         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   76C    P4             43W /  100W |     643MiB /   8151MiB |     14%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            7928    C+G   ...ng\Sysdiag\bin\HipsDaemon.exe      N/A      |
|    0   N/A  N/A           15940      C   ...Files\deeplearning\python.exe      N/A      |
+-----------------------------------------------------------------------------------------+
---- compute apps ----
pid, process_name, used_gpu_memory [MiB]
7928, E:\Program Files\Huorong\Sysdiag\bin\HipsDaemon.exe, [N/A]
15940, D:\ProgramFiles\deeplearning\python.exe, [N/A]
```

**快照要点**：

| 观察项 | 快照 1（20:39:17） | 快照 2（20:39:37） |
|---|---|---|
| GPU-Util | 12% | 14% |
| 显存 | 643 MiB / 8151 MiB | 643 MiB / 8151 MiB |
| 温度 / 功耗 | 79C / 37W（cap 100W） | 76C / 43W（cap 100W） |
| Perf 状态 | P4 | P4 |
| 训练进程 | PID 15940（python.exe，见 compute apps） | 同左 |
| 噪声进程 | PID 7928 HipsDaemon.exe（火绒防护，C+G） | 同左 |

显存 643 MiB 距任务所述 7.5 GB 告警线（8151 MiB 的 ~92%）不到 8%，**无显存压力**。

### 5.2 1s 粒度 GPU/CPU 监控统计（monitor.csv）

来源：正式基准运行期间（20:27:59 → 20:29:51），103 个样本、窗口 112 s。列：`ts, gpu_util_pct, gpu_memutil_pct, gpu_mem_used_mib, gpu_temp_c, gpu_power_w, py_cpu_s, py_ws_mb`。

| 指标 | avg | min | max |
|---|---:|---:|---:|
| GPU 利用率 (%) | 29.5 | 0 | 70 |
| GPU 显存带宽利用率 (%) | 9.9 | 0 | 29 |
| 显存占用 (MiB) | 618.73 | 46 | 643 |
| 温度 (℃) | 75.46 | 59 | 83 |
| 功耗 (W) | 39.08 | 5.72 | 63.97 |
| python 累计 CPU (s) | 首 35 → 末 134.2（Δ99.2 s / 112 s ≈ **0.89 核**） | — | — |
| python 工作集 (MB) | 7830 | — | 8014.4 |

说明：
- 首末样本：`20:27:59,0,0,46,59,5.72,35,2110.9` … `20:29:51,0,0,643,74,13.49,134.2,3775.6`（末样本 util 已归 0，进入收尾，故 min 含 0）。
- `py_cpu_s` 为跨 python 进程累计 CPU 秒求和；窗口内 max（182 s）高于末值（134.2 s），因中途曾同时统计到多个 python 进程、进程退出后求和回落；0.89 核由首末差值得出。
- `py_ws_mb`（≈7.8–8.0 GB）是进程工作集口径（含 CUDA 映射/共享页），**与显存无关**，不能读作显存压力。

---

## 6. 分相耗时与吞吐口径（TRAIN_PROF=1）

第二次分相跑：同参 16384 步（32 迭代）、`TRAIN_PROF=1`，`--out` 指向 temp 目录避免新增仓库产物。解析 `bench-prof.log`（rows=21，全部 match，无空值），稳定段均值：

| 分项 | 均值 | 含义 |
|---|---:|---|
| collect | 1.057 s | 每迭代采集相总耗时 |
| └ step | 0.2 s | 引擎 `envs.step()` 环境推进（8 envs × 64 步） |
| └ fwd | 0.857 s | 采集相内策略前向/编码（= collect − step；日志 fwd 位因 0.1 s 舍入显示 0.89 s） |
| update | 0.614 s | PPO 模型更新相 |
| 合计 | 1.671 s | collect + update（= 端到端单迭代耗时） |
| c-sps（报数） | 491.6 | 采集相 samples/s |

**占比**（以合计 1.671 s 为分母；分项按 0.1 s 粒度记录、存在舍入）：

| 相 | 占比 |
|---|---:|
| step（引擎步进） | 12.0% |
| fwd（策略前向/编码） | 51.3% |
| update（模型更新） | 36.7% |

**派生速率**（batch = 512 env steps）：

| 口径 | 计算 | 速率 |
|---|---|---:|
| 纯引擎 env steps/s | 512 / 0.2 s | ≈ 2560 env-steps/s |
| 采集相 samples/s | 512 / 1.057 s | ≈ 484 samples/s |
| 更新相 samples/s（Training throughput） | 512 / 0.614 s | ≈ 834 samples/s |
| 端到端 samples/s（Environment steps/s） | 512 / 1.671 s | ≈ 306 samples/s（与日志 sps 一致） |

**两种口径的区分（避免重复计算 num_envs）**：
- **Environment steps/s（对局采样）**：端到端 306 env-steps/s（主口径 307.67 即此口径的日志值）——每迭代推进 512 个 env steps（= 8 并行环境 × 64 步），`num_envs` 已包含在 batch 里，不得再乘一次。
- **Training throughput（模型更新）**：update 相 834 samples/s——只覆盖 PPO 更新，不含采集等待；它不是对局采样速率，两者不可混用。

---

## 7. 训练时长估算（三口径 + 文档参考）

| steps/s 口径 | 来源 | 10M | 100M | 1B | 11B |
|---:|---|---:|---:|---:|---:|
| **307.67** | **主口径：it=10–30 sps 均值** | **9.0 h** | **3.8 d** | **37.6 d** | **413.8 d** |
| 305 | 末值 avg（全进程累计） | 9.1 h | 3.8 d | 37.9 d | 417.4 d |
| 302 | it=10–30 avg 列均值 | 9.2 h | 3.8 d | 38.3 d | 421.6 d |
| 200 | 任务文档参考值（对照） | 13.9 h | 5.8 d | 57.9 d | 636.6 d |

换算公式：小时 = steps / (sps × 3600)；天 = 小时 / 24。11B（11,000,000,000 env steps）在主口径下 = 11e9 / 307.67 / 86400 ≈ **413.8 天**，为主要风险项。

---

## 8. 瓶颈判断

| 证据 | 数值 | 判断 |
|---|---|---|
| GPU 利用率（1s 采样均值） | 29.5%（瞬时快照 12–14%） | GPU 未饱和 |
| python CPU 占用 | ≈ 0.89 核（远低于饱和） | CPU 未饱和 |
| 显存占用 | 峰值 643 MiB（<8% of 8151 MiB） | 无显存瓶颈 |
| 单迭代结构 | step 12.0% / fwd 51.3% / update 36.7%，顺序执行 | 时长由「采集前向 + 小批次编排」主导 |

**结论**：本负载既非 GPU 密集瓶颈，也非 CPU 密集瓶颈（任务预案中的「GPU 低 CPU 高 → 环境模拟/Python 瓶颈」不成立——CPU 同样不高）。总时长由训练循环的**顺序结构**主导：每迭代 63.3% 时间花在采集相（其中前向/编码 51.3%、引擎步进 12.0%），36.7% 花在 PPO 更新相；GPU 在采集相仅做 batch=512 的小规模前向，故利用率低。

**潜在优化方向（本次未启用、未验证，仅作参考）**：增加并行环境数（num_envs↑，提升 GPU 前向批量）、`--bf16-collect`、采集相 `torch.compile`、`--async-collect`（采集与更新重叠）。启用任一前建议先复核显存与正确性。另：噪声进程 HipsDaemon.exe（火绒）持续占用少量 GPU，可考虑加白名单。

---

## 9. 交付说明与效力标注

### 9.1 仓库改动（唯一一处）

- `rl/train_selfplay.py`：**+7 行**，为 HEAD（`4fbf2c5`）真实缺陷的最小修复——补齐缺失导入；`GENERATED` 改为 try/except 置 `{}`；`gen` / `all+gen` 解码模式显式 `SystemExit` 报错。未提交。

### 9.2 训练依赖与产物

- `agent/deck.csv`：从 wheel 内嵌样例 deck 重建（60 行、LF），读回校验通过；训练依赖（`--decks sample` 需要该文件）。
- `runs/bench-rtx5060/ckpt_32768.pt`：正式基准末迭代检查点（未跟踪，仓库工作树新增产物）。
- 临时产物（均在 `C:\Users\Swcmb\AppData\Local\Temp\opencode\`，不落仓库）：引擎 shim、`bench-gpu.log`、`bench-prof.log`、`nvidia-smi-midrun.txt`、`monitor.csv`、`bench-smi2.log` 及复现脚本 `bench-dl.ps1` / `gpu-monitor.ps1` / `bench-prof.ps1` / `bench-smi.ps1` / `bench-smi2.ps1` / `parse-logs.ps1`。

### 9.3 git 状态快照（报告撰写时）

```text
HEAD: 4fbf2c5
 M rl/train_selfplay.py
?? AGENT.md
?? Prompt/
?? agent/
?? rules/
?? runs/
```

说明：`AGENT.md`、`rules/` 非本次基准产物（先于本次任务存在）；`Prompt/` 含任务文件与本报告；`agent/`、`runs/` 见 §9.2。

### 9.4 效力标注（必读）

1. **JSON 路径效力**：本机自研 native 编码链路当前**不可用**——wheel 自带 CABT 缺失导出符号 `GetBinaryObs`、`AdvanceLog`、`WouldKO`（不在本次范围）。本次全部数字均为**公开引擎 JSON 路径**（`native_encode=False`）的实测值；native 链补齐后应复测。
2. **WSL2 偏差**：任务建议 WSL2，实测为 Windows 原生 conda（§2），steps/s 口径不受影响。
3. **时长估算的边界**：以上估算基于当前配置（num_envs=8、batch=512、未 compile、JSON 路径）线性外推；优化项启用或配置改变后需按新基线重估。
4. 主口径 307.67 与 `avg` 终值 305 差异 <1%，不影响量级判断。
