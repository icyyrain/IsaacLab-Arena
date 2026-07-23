# Windows Native PPO Demo 快速说明

这份 worktree 用于先验证“Isaac Lab 仿真 + 强化学习策略 + 视频输出”链路，再逐步加入 HIL-SERL、人工干预和自定义插接任务。当前 PPO demo 使用 Factory 的 `Isaac-Factory-PegInsert-Direct-v0`；它不是 USB 最终任务，也还不是 HIL-SERL。

## Worktree 与目录

- 原项目：`C:\Projects\isaac`
- 本 worktree：`C:\Projects\isaac-hil-serl`
- 分支：`icyyrain/feature/hil-serl-sim-validation`

两个目录共享 Git 对象库，但检出的分支和工作文件彼此独立。后续 HIL-SERL 实验应在本 worktree 中运行。

| 路径 | 用途 |
| --- | --- |
| `IsaacLab_HIL_SERL_Simulation_Validation_Plan.md` | 分阶段实现方案 |
| `submodules/IsaacLab/` | Isaac Lab 源码、Factory 环境和 RL-Games 配置 |
| `patches/isaaclab/windows-native.patch` | Windows native 兼容补丁 |
| `tools/apply_windows_native_isaaclab_patch.ps1` | 检查版本并应用补丁 |
| `tools/run_windows_factory_ppo_demo.ps1` | 一键播放 PPO、录视频并检查机械臂确实运动 |
| `tools/play_factory_ppo_with_camera.py` | PPO player 和近距离相机实现 |
| `logs/factory_ppo_demo/` | 生成的视频；不提交 Git |

## 已验证环境

这是当前机器已跑通的组合，不是 NVIDIA 的正式兼容矩阵：

- Windows 11、NVIDIA RTX GPU（本机为 RTX 5090 D v2）
- Python `3.11.15`
- Isaac Sim `5.1.0.0`
- Isaac Lab submodule commit `55df2c34390ba94b22d41879514c5485c5115462`（约为 `v3.0.0-beta`）
- PyTorch `2.10.0+cu128`
- NumPy `1.26.0`
- Gymnasium `1.2.1`
- RL-Games `1.6.1`
- opencv-python-headless `4.11.0.86`

至少准备 32 GB 内存、足够的 SSD 空间和可访问 NVIDIA 资产服务器的网络。首次启动会下载 USD/材质并建立 shader cache，可能需要 10–30 分钟；后续通常会快很多。Windows 建议开启 long paths。

## 配置 Windows native 环境

当前脚本固定使用 `C:\Isaac\envs\arena-py311\python.exe`。若该环境已经存在且版本一致，不要重复安装。全新机器可从 Miniconda PowerShell 开始：

```powershell
conda create --prefix C:\Isaac\envs\arena-py311 python=3.11.15 -y
conda activate C:\Isaac\envs\arena-py311
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

Set-Location C:\Projects\isaac-hil-serl\submodules\IsaacLab
.\isaaclab.bat -i rl_games

python -m pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy==1.26.0" "gymnasium==1.2.1" "rl-games==1.6.1" "opencv-python-headless==4.11.0.86"
```

然后在 worktree 根目录初始化 submodule 并应用四项兼容修复（RTX 扩展过滤、Windows 文件锁、PhysX 初始化重试、Warp tensor view fallback）：

```powershell
Set-Location C:\Projects\isaac-hil-serl
git submodule update --init -- submodules/IsaacLab
.\tools\apply_windows_native_isaaclab_patch.ps1
```

脚本可重复执行，并会拒绝给错误的 Isaac Lab commit 打补丁。补丁应用后 `git status` 显示 `submodules/IsaacLab` 为 modified 是预期现象；不要提交这个 detached submodule 状态。

> 上游 Isaac Lab 源码当前面向 Isaac Sim 6.0，而本机 native runtime 是 5.1.0，所以必须保留上述补丁。项目常规 Arena 测试仍以 Docker 为准；这里的 PPO 路径不需要 Docker Desktop/WSL2。

## 下载官方 PPO checkpoint

该模型是 NVIDIA Isaac Lab 发布的 RL-Games PPO checkpoint，约 203 MiB。其官方资产路径由 Isaac Lab 的 `get_published_pretrained_checkpoint("rl_games", task)` 生成；下面的 URL 与本 checkout 的 `apps/isaaclab.python.kit` 配置一致。

```powershell
Set-Location C:\Projects\isaac-hil-serl

$checkpointDir = ".\submodules\IsaacLab\.pretrained_checkpoints\rl_games\Isaac-Factory-PegInsert-Direct-v0\Assets\Isaac\6.0\Isaac\IsaacLab\PretrainedCheckpoints\rl_games\Isaac-Factory-PegInsert-Direct-v0"
$checkpoint = Join-Path $checkpointDir "checkpoint.pth"
New-Item -ItemType Directory -Force $checkpointDir | Out-Null

if (-not (Test-Path $checkpoint)) {
    Invoke-WebRequest `
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rl_games/Isaac-Factory-PegInsert-Direct-v0/checkpoint.pth" `
        -OutFile $checkpoint
}
```

原 checkpoint 由 NumPy 2.x 保存，本地 NumPy 1.26 不能直接反序列化。demo 脚本首次运行时会自动生成旁边的 `checkpoint-numpy1.pth`，不会修改官方下载文件。

## 运行 PPO 视频 demo

```powershell
Set-Location C:\Projects\isaac-hil-serl
.\tools\run_windows_factory_ppo_demo.ps1 -VideoLength 150 -NumEnvs 1 -Seed 0
```

输出为：

```text
C:\Projects\isaac-hil-serl\logs\factory_ppo_demo\factory-ppo-peg-insert.mp4
```

脚本以 headless 模式运行，但会创建离屏相机、录制 MP4，并检查帧数和画面运动；`Seed 0` 的已验证运行在约第 54 步触发过成功。需要更长视频可将 `-VideoLength` 改为 `600`。这一步只是播放现成 PPO 策略，用来证明环境、策略、PhysX 和录像链路正常；之后才进入 HIL-SERL replay buffer、SAC learner 和人工干预实现。

参考：[Isaac Sim 5.1 pip 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)、[Isaac Lab pip 安装](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)。
