# Isaac Lab 中的 HIL-SERL 仿真验证计划

## 1. 研究目标

本项目的核心不是复刻某一种 USB、Factory 或 AutoMate 几何，而是回答下面的问题：

> 在接触丰富的机械臂插入任务中，初始人工示范和在线人工干预能否让 off-policy 强化学习更快、更稳定地学到高成功率策略？

任务环境只是验证 HIL-SERL 的实验载体。第一阶段固定使用一个成熟、可训练的插入环境，避免同时修改机器人资产、碰撞几何、控制器和学习算法。

最终证据必须来自关闭人工干预后的自主评测，而不是训练期间由人接管完成的成功率。

## 2. 当前范围

### 2.1 第一阶段包含

- Isaac Lab 仿真和 Franka Panda。
- `Isaac-Factory-PegInsert-Direct-v0` 作为固定插入任务。
- 6D 末端执行器增量或 twist 动作，夹爪初期保持闭合。
- state-based observation。
- 仿真 ground-truth success reward。
- off-policy actor-critic learner。
- replay buffer、demo buffer 和 intervention 数据。
- SpaceMouse 人工示范与在线接管。
- 可重复的 scripted expert/intervention，用于自动化回归测试。
- 三组核心消融实验和统一评测。

### 2.2 第一阶段明确不做

- 自定义 USB USD 资产。
- Factory 到 AutoMate 的任务迁移。
- 抓取线缆或从桌面拾取插头。
- 视觉策略和视觉 reward classifier。
- sim-to-real 或真实机械臂部署。
- 多种 embodiment、scene 或 task 的统一 benchmark。

这些内容只有在 HIL-SERL 主实验得到可靠结果后才进入后续阶段。

## 3. 核心假设

### H1：Demonstration 提高早期样本效率

在相同环境步数预算下，加入少量成功示范的 SAC 应比纯在线 SAC 更早获得非零成功率，并更快达到目标成功率。

### H2：Intervention 改善困难状态的数据覆盖

在 SAC + demo 的基础上，在线 corrective intervention 应减少连续失败，向 replay buffer 增加从失败边缘恢复的有效 transition。

### H3：HIL 的收益在 autonomous evaluation 中仍然存在

训练结束并关闭 intervention 后，HIL-SERL 组应在相同随机化分布上保持更高成功率或使用更少环境步数达到同等成功率。

## 4. 实验组

### E0：官方 PPO smoke baseline

用途仅限于验证：

- 仿真、资产和物理场景能启动；
- Factory task 的 observation/action/termination 正常；
- 官方 checkpoint 可以加载并产生合理插入行为。

PPO 不参与 HIL-SERL 样本效率结论，因为它是 on-policy 算法，数据使用方式与 SAC 不同。

### E1：Online SAC

```text
随机初始化 SAC
  -> 环境交互
  -> online replay buffer
  -> SAC update
```

不使用 demonstration，不允许 intervention。这是主要的 off-policy 对照组。

### E2：SAC + Demonstrations

```text
demo buffer + online replay buffer
              -> mixed sampling
              -> SAC update
```

训练开始前加入固定数量的成功示范。所有随机种子使用相同的一份 demo 数据集，避免把示范质量差异误认为算法差异。

### E3：SAC + Demonstrations + Interventions

```text
policy action ----+
                  +--> action switch --> executed action --> environment
human action -----+

demo data + policy data + intervention data
                  -> replay sampling
                  -> SAC update
```

这是完整的 HIL-SERL 风格实验组。训练期间记录 intervention 次数、持续步数和人工操作时间。

## 5. 公平对比约束

E1、E2、E3 必须保持下列条件一致：

- 相同任务配置、控制频率和 action scaling；
- 相同 observation、reward 和 termination；
- 相同 actor/critic 网络和优化器；
- 相同训练环境数量；
- 相同最大环境 transition 数；
- 相同 update-to-data ratio；
- 相同评测随机种子集合；
- 相同 autonomous evaluation episode 数；
- 至少 3 个训练随机种子，条件允许时使用 5 个。

仿真能够轻易生成大量并行数据。为保留 HIL-SERL 的样本效率研究意义，核心实验不能让某一组单独使用大规模并行环境。建议先使用 `num_envs=1` 打通 HIL，再选择固定的小规模并行数完成正式消融。

除了 environment steps，还必须单独报告 demonstration transitions 和 intervention transitions，不能把人工数据当成免费数据。

## 6. 环境、动作、观测和奖励

### 6.1 固定环境

第一阶段使用：

```text
Isaac-Factory-PegInsert-Direct-v0
```

第一阶段不修改 Factory 几何。只有在它无法支持必要的 teleoperation 或 transition 记录时，才通过外部 wrapper 做最小适配，不直接修改 IsaacLab 子模块。

### 6.2 动作

Policy、scripted expert 和 SpaceMouse 必须输出同一种动作：

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

动作经过统一的 clipping、scaling 和控制器映射。系统同时保存 policy proposal 与真正执行的 action。

### 6.3 V1 observation

首版使用 state observation，例如：

```text
joint position / velocity
end-effector pose / velocity
plug pose
socket pose
plug-to-socket relative pose
```

首版目标是验证学习与 HIL 数据机制，不验证视觉表征能力。

### 6.4 V1 reward

首版使用仿真真值构造稀疏成功奖励，并在必要时保留 Factory 官方 shaping reward 作为单独配置。主实验不得在不同实验组之间使用不同 reward。

需要明确记录：

- success 判定；
- insertion depth；
- 横向和姿态误差；
- timeout 与失败 termination。

## 7. Transition 数据协议

所有数据源使用同一个 schema：

```python
{
    "observation": ...,
    "policy_action": ...,
    "executed_action": ...,
    "next_observation": ...,
    "reward": ...,
    "terminated": ...,
    "truncated": ...,
    "source": "policy" | "demonstration" | "intervention",
    "intervention": True | False,
    "episode_id": ...,
    "step_id": ...,
}
```

SAC 的 Bellman update 使用 `executed_action`。`policy_action` 仅用于分析人在什么状态否决了策略以及策略与人工动作的差异。

数据写入前验证：

- shape 和 dtype；
- NaN/Inf；
- action 范围；
- episode 边界；
- success label；
- reset 前后的 transition 不串联。

## 8. Intervention 设计

### 8.1 Scripted intervention

先实现确定性的 scripted expert/intervention，用于：

- 自动化测试 action switch；
- 生成格式正确的少量示范；
- 重复运行算法消融，降低不同操作者造成的方差；
- 在无人值守训练前验证 replay 数据正确。

它是工程和可重复性工具，不能替代最终的人类 HIL 实验。

### 8.2 Human intervention

使用 SpaceMouse：

- 默认由 policy 控制；
- 操作者触发 takeover 后执行 human action；
- takeover 结束后平滑交还 policy；
- reset、急停和接管必须使用独立按键；
- UI 显示当前控制权、episode 成功状态和累计干预步数。

训练过程中逐渐减少 intervention 可以作为协议的一部分，但减少规则必须在实验前固定，不能根据某次结果临时调整。

## 9. 软件架构

### 9.1 官方 HIL-SERL 是算法基准

算法基准固定为 `rail-berkeley/hil-serl` commit：

```text
c32939bccb65f3b8c43a9f9add3d322d4ab0264a
```

优先直接复用官方 `serl_launcher`、SAC/RLPD agent、replay/data store、Agentlace actor/learner 通信、demo/intervention 混合采样和 checkpoint 逻辑。不得重新实现一个简化 SAC 并将其称为 HIL-SERL。若必须修改上游代码，应保持 Apache-2.0 许可和来源说明，并把改动限制为明确的兼容层。

第一版不复用真实 Franka ROS server、硬件控制器、相机驱动和任务专属 reset。仿真 ground-truth success 暂时代替视觉 reward classifier；因此第一版验证的是 HIL-SERL 的 off-policy learning、demonstration 和 intervention 核心，而不是完整的视觉系统。

### 9.2 Windows/WSL2 进程边界

Isaac Sim 环境和官方 HIL-SERL 使用隔离的 Python 环境：

```text
Windows native (Python 3.11)
Isaac Sim + Factory + SpaceMouse + environment RPC server
                         ^
                         | observation/action/reset
                         v
WSL2 (Python 3.10)
official HIL-SERL actor <-> official HIL-SERL learner
              transitions -> | <- parameter synchronization
```

这样做有两个必要原因：

- 已验证的 Isaac 环境使用 Gymnasium 1.2.1，官方 HIL-SERL 固定 Gymnasium 0.29.1，不能在同一环境中直接安装；
- JAX 不支持原生 Windows NVIDIA GPU，learner 应在 WSL2/Linux 使用 CUDA。WSL2 不运行 Isaac Sim，因此不依赖 Vulkan/PhysX 图形接口。

长期功能放在新的顶层 integration package 中，不修改 `submodules/IsaacLab`，也不复制官方 learner：

```text
isaaclab_hil_serl/
├── config/
├── isaaclab_hil_serl/
│   ├── envs/
│   │   ├── factory_adapter.py
│   │   └── factory_rpc_server.py
│   ├── protocol/
│   │   ├── messages.py
│   │   └── remote_env.py
│   ├── teleop/
│   │   ├── intervention_manager.py
│   │   └── spacemouse.py
│   ├── data/
│   │   └── validation.py
│   └── evaluation/
│       └── autonomous_eval.py
├── scripts/
│   ├── run_windows_env_server.py
│   ├── collect_demos.py
│   ├── train_hil_serl.py
│   └── evaluate.py
├── tests/
└── pyproject.toml
```

## 10. 指标

### 10.1 主要指标

- autonomous success rate；
- 达到 50%、80%、90% success 所需 environment transitions；
- 固定 transition budget 下的 success rate；
- 学习曲线的均值、方差和置信区间。

### 10.2 人工成本

- demonstration episode 数和 transition 数；
- intervention event 数；
- intervention transition 数；
- 人工接管总时长；
- 每次成功平均需要的人工时间。

### 10.3 辅助指标

- episode length；
- insertion depth；
- position/orientation error；
- intervention 后恢复成功率；
- 不同初始位置和姿态扰动下的成功率。

## 11. 实施里程碑

### M0：环境 smoke test

- 初始化独立 worktree、IsaacLab 子模块和开发容器；
- 运行 Factory 官方 PPO checkpoint；
- 确认环境可以 reset、step、render 和 terminate。

完成标准：官方策略能在本机启动并产生合理 rollout，或者得到有日志支持的明确兼容性问题。

### M1：环境适配与数据协议

**状态：已完成（2026-07-23）。**

- 在隔离环境中安装固定 commit 的官方 HIL-SERL，并完成最小 actor/learner smoke test；
- 创建 `isaaclab_hil_serl` extension；
- 实现统一 action adapter；
- 实现 Windows environment RPC server 和 WSL2 remote Gym env；
- 实现 transition schema 和持久化；
- 添加 action switch、episode boundary 和 dataset round-trip 测试。

完成标准：随机策略、scripted action 和人工 action 都能生成可验证的数据集，并且 WSL2 actor 能 reset/step Windows Factory 环境。

### M2：SAC baseline

- 接入固定版本的官方 HIL-SERL SAC/RLPD learner；
- 完成 E1；
- 固定网络、normalization、UTD 和 seed 配置；
- 输出 autonomous evaluation 和学习曲线。

完成标准：SAC 可以稳定更新；若不能解决任务，能通过 Q 值、loss、reward 和数据覆盖诊断原因。

### M3：Demonstration ablation

- 采集或生成固定 demo dataset；
- 实现 demo/online 混合采样；
- 完成 E2 与 E1 对照。

完成标准：多随机种子结果可以回答 H1。

### M4：Intervention ablation

- scripted intervention 自动化验证；
- SpaceMouse takeover；
- 完成 E3 与 E2 对照；
- 关闭人工控制后统一评测。

完成标准：结果可以回答 H2 和 H3，同时报告人工成本。

### M5：可选扩展

只有 M4 得到稳定结果后才依次考虑：

1. AutoMate 的非对称 plug/socket；
2. 自定义 USB 插入几何；
3. RGB/wrist camera policy；
4. binary visual reward classifier；
5. sim-to-real。

## 12. 决策门槛

进入下一阶段前满足：

- E1/E2/E3 使用完全相同的评测代码；
- 至少 3 个随机种子完成；
- 训练期间和评测期间的数据严格分开；
- 评测关闭 intervention；
- 所有人工数据成本均被统计；
- 结果可由保存的 config、seed、checkpoint 和 dataset 复现。

如果 E3 没有优于 E2，优先检查 intervention 数据质量、动作坐标系、buffer sampling 和 credit assignment，而不是立刻更换 USB 几何。

## 13. 一句话总结

> 固定一个成熟的 Franka 插入环境，以 Online SAC、SAC + Demo、SAC + Demo + Intervention 三组公平消融为主线，用自主成功率、样本效率和人工成本验证 HIL-SERL；Factory、AutoMate 和 USB 只代表不同难度的实验载体。

## 14. 当前执行状态（2026-07-23）

- 独立 worktree：`C:\Projects\isaac-hil-serl`。
- 分支：`icyyrain/feature/hil-serl-sim-validation`，基于 `origin/main`。
- IsaacLab 子模块固定在主仓库记录的 `55df2c3`；Windows-native 运行使用已单独提交的兼容补丁，不新增本任务专属的子模块改动。
- 独立容器 `isaaclab_arena-latest-isaac-hil-serl` 可以导入 Arena package 并检测 RTX 5090 CUDA compute，但 Docker Desktop/WSL2 没有向 Isaac Sim 提供可用的 Vulkan/PhysX 图形接口，因此不作为本机 PPO 运行路径。
- Factory 官方 PPO checkpoint 已从 Isaac Sim 6.0 资产服务器成功下载，大小约 203 MiB。
- 本机 Windows-native Isaac Sim 5.1 已成功运行当前 IsaacLab 子模块的 Factory 环境、PhysX GPU step 和 RL-Games PPO policy。
- 官方 checkpoint 的 NumPy 2 pickle 路径与本机 NumPy 1.26 不兼容；运行脚本会生成本地兼容副本，不修改原 checkpoint。
- seed 0 的单环境 PPO rollout 在第 54 个 environment step 触发 Factory 官方插入成功判定。
- 已生成 1280×720、15 FPS 的可视化插入视频；录制路径显式执行 Fabric 到 RTX 场景的姿态同步，画面可见机械臂移动、对准和插入。
- `tools/run_windows_factory_ppo_demo.ps1` 提供可复现入口，并通过显式离屏相机和 Kit render update 适配本机无 GUI 录制；运行结束后自动检查非空帧和显著帧间运动，静止视频会判定失败。
- 官方 `rail-berkeley/hil-serl` 已作为 `third_party/hil-serl` submodule 固定到 commit `c32939b`。
- WSL2 Ubuntu 22.04、Python 3.10.12 和 RTX 5090 CUDA 映射均可用。
- 官方 HIL-SERL 的 Gymnasium 0.29.1 与 Windows Isaac 环境的 1.2.1 冲突，因此确定采用 Windows environment server + WSL2 official actor/learner 的隔离架构。
- WSL2 隔离环境 `/root/.venvs/hil-serl-c32939b` 已验证 JAX 0.4.35 CUDA、官方 `SACAgent`、`ReplayBuffer` 和 Agentlace `TrainerServer/TrainerClient` 均可导入运行。
- 上游 open-ended requirements 在 2026 年会错误升级 JAX/OpenCV；已将验证通过的关键版本记录在 `requirements/hil-serl-wsl-constraints.txt`。
- `isaaclab_hil_serl.protocol` 已实现无 pickle 的 length-prefixed JSON/NumPy RPC、Windows environment server 和 WSL2 `RemoteEnv`。
- `Isaac-Factory-PegInsert-Direct-v0` 已接入 RPC：wire observation 为固定字段顺序的 43 维 `float32` state，action 为归一化的 6D EEF delta。
- transition schema 已区分 `policy_action` 与 `executed_action`，处理 success terminal 与 timeout truncation，并支持不使用 pickle 的 NPZ 持久化、episode boundary 校验和 dataset round-trip。
- host-side adapter、RPC、collector、SpaceMouse provider 和数据协议测试为 16/16 通过。
- 真实链路已经跑通：WSL2 官方 `SACAgent` 在 RTX 5090 上采样动作，Windows Factory 执行 PhysX step，transition 回到 WSL2 后完成官方 actor、critic 和 temperature 的一次 GPU 梯度更新。
- scripted action switch 已在真实链路验证：策略 proposal 与脚本动作不同时，Factory 执行脚本动作，`info["intervene_action"]` 和 SERL Bellman transition 均保存实际执行动作。
- 多步 collector 已从真实 Factory 采集 5 条 transition（含 1 条 scripted intervention），保存为无 pickle NPZ 后在 WSL2 完整重载并重新校验；smoke 数据位于忽略提交的 `logs/hil_serl_smoke/`。
- 官方 Agentlace 最小 actor/learner 网络链路已通过：`QueuedDataStore` 上传 5/5 条 transition，learner replay 完成一次官方 SAC GPU update，并把新参数广播回 actor callback。
- Windows server 已接入 Isaac Lab 原生 `Se3SpaceMouse`：运动超过 deadzone 时接管，回中时释放，人工动作经过统一归一化、clipping、action switch 和 transition schema；fake-device 回归已验证人工动作可保存、重载为合法 NPZ 数据集。
- 当前主机未连接 SpaceMouse，因此本轮不能执行真实 HID 人手移动；接入代码可用 `run_windows_factory_rpc_server.ps1 -InterventionDevice spacemouse` 直接启用，真实操作员实验属于 M4 的 intervention ablation。
- 独立进程回归已通过：Windows Factory、WSL2 actor 和 WSL2 learner 三进程运行，8/8 条 transition 上传到 replay，完成 3 次官方 SAC GPU update，actor 收到更新后的第二个参数版本；其中 1 条 scripted intervention 被保存在数据集中。

M0（官方 PPO smoke baseline）和 M1（环境适配与数据协议）均已完成。下一步进入 M2，固定 E1 Online SAC 的训练配置并运行第一轮持续训练。
