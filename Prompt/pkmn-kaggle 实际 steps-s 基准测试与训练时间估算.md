测 `pkmn-kaggle` 的实际 `steps/s`，最可靠的方法是：用项目自带的训练入口运行一个短基准，记录实际完成的环境步数与耗时，同时观察 GPU、CPU、RAM 和显存占用。

需要先区分两个指标：

- Environment steps/s：每秒完成多少个模拟环境步，主要衡量对局采样速度。
- Training throughput：每秒处理多少训练样本或执行多少次模型更新，衡量训练整体速度。

估算 1,000 万步、1 亿步训练时，必须使用项目中 `num_envs` 和 `steps` 的实际定义，避免把并行环境数重复计算。

## 1. 直接运行基准测试

`pkmn-kaggle` 的 `rl/train_selfplay.py` 已经内置 `sps` 和 `avg` 指标，不需要修改源码。` sps` 是当前迭代的速度，`avg` 是从本次启动以来的平均速度。源码：[rl/train_selfplay.py](https://github.com/Michal1337/pkmn-kaggle/blob/main/rl/train_selfplay.py?utm_source=chatgpt.com)

建议在 WSL2 Ubuntu 中运行，前提是你已按仓库 README 安装依赖并配置好 CABT 引擎。

```
cd ~/pkmn-kaggle

PYTHONPATH=. python -m rl.train_selfplay \
  --decks sample \
  --num-envs 8 \
  --num-steps 64 \
  --num-minibatches 16 \
  --total-timesteps 32768 \
  --log-every 1 \
  --out runs/bench-rtx5060
```

这是一组用于初步测量的参数：

- 8 个并行环境，每轮采集 64 步。
- 每次迭代采集 \\(8 \times 64 = 512\\) 个环境步。
- 总预算 32,768 步，约 64 次迭代。
- 使用 `sample` 牌组，减少牌组池的额外变量。
- 不启用 `torch.compile`，先测量基础训练路径。

如果出现显存不足、引擎初始化失败或缺少原生依赖，先解决启动问题，再记录速度。

## 2. 查看实际速度

训练日志中会出现类似格式：

```
step=512 it=1/64 sps=... avg=...
step=1024 it=2/64 sps=... avg=...
```

这里的数字只是格式示例，并非你的实测结果。

| 指标     | 含义             | 怎么用       |
| ------ | -------------- | --------- |
| `sps`  | 当前迭代的环境步/秒     | 观察稳定阶段的速度 |
| `avg`  | 本次启动以来的平均环境步/秒 | 估算整体耗时    |
| `step` | 累积环境步数         | 确认实际进度    |

记录第 10–30 次迭代的 `sps` 和 `avg`。 如果速度逐渐稳定，就用稳定阶段的 `avg` 估算训练时间。首次运行的初始化、CUDA 内核加载等开销会影响早期数据。

## 3. 同时监控显卡和内存

在另一个 WSL2 终端运行：

```
nvidia-smi -l 1
```

每秒查看 GPU 利用率、显存使用量、温度和功耗。

再开一个终端监控 CPU 和 RAM：

```
watch -n 1 'free -h; echo; ps -eo pid,pcpu,pmem,rss,comm --sort=-pcpu | head -15'
```

如果 GPU 利用率较低而 CPU 使用率很高，环境模拟或 Python 进程可能是瓶颈；如果显存接近 7.5 GB，则需要降低并行环境数或训练批次。

## 4. 计算训练需要多久

实测得到稳定速度后，按下面的公式计算：

\\[ \text{训练小时数} = \frac{\text{目标环境步数}}{\text{实测 steps/s}\times3600} \\]

例如，如果实测稳定速度为 200 steps/s：

| 目标步数    | 理论耗时    |
| ------- | ------- |
| 1,000 万 | 13.9 小时 |
| 1 亿     | 5.8 天   |
| 10 亿    | 57.9 天  |
| 110 亿   | 636.6 天 |

这些是按持续运行计算的理论值，实际完整训练还会受到模型更新、保存检查点和评估的影响。

先运行上面的 32,768 步基准，把最后 10 条训练日志和 `nvidia-smi` 输出发来，就能据此判断你的 RTX 5060 是受 GPU、CPU 还是环境采样限制，并计算更可信的训练时间。