# Windows 原生部署 IsaacLab-Arena + OpenPI 演示

本文记录在 Windows 原生 Isaac Sim 上跑通 IsaacLab-Arena OpenPI demo 的部署方式，供另一台 Windows 机器上的 Codex 复现。

已验证环境：

- Windows 11
- NVIDIA RTX 4070 Ti 12GB
- Windows 原生 Python 环境：`C:\Isaac\envs\arena-py311`
- Isaac Sim pip 包：`isaacsim[all,extscache]==5.1.0.0`
- Isaac Lab：`isaaclab==4.5.24`
- Torch：`2.7.0+cu128`
- OpenPI server：WSL2 Ubuntu + Docker image `isaaclab_arena_openpi-server:latest`

核心结论：Isaac Sim/Isaac Lab 在 Windows 原生 Python 里运行，OpenPI 推理服务在 WSL2 Docker 里运行。不要把 Isaac Sim 放进当前 WSL2/Docker 跑，因为这台机器的 WSL Vulkan 只暴露了 llvmpipe CPU Vulkan，没有 NVIDIA Vulkan/RTX 渲染通路。

## 1. 先决条件

Windows 侧：

- NVIDIA 驱动可用，PowerShell 中 `nvidia-smi` 能看到 GPU。
- 安装 Git、Git LFS、Miniconda 或 Anaconda。
- 建议使用短路径，例如 `C:\Isaac`，避免 Windows 路径长度问题。
- NVIDIA Isaac Sim EULA 需要接受，运行前设置：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "Y"
```

WSL/Docker 侧：

- 安装 WSL2 Ubuntu。
- 安装 Docker Desktop，并启用 WSL integration。
- WSL 中能使用 GPU：

```powershell
wsl -d Ubuntu -- nvidia-smi
```

Docker 中能使用 GPU：

```powershell
wsl -d Ubuntu -- docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. 克隆仓库

```powershell
New-Item -ItemType Directory -Force -Path C:\Isaac | Out-Null
Set-Location C:\Isaac
git clone https://github.com/isaac-sim/IsaacLab-Arena.git
Set-Location C:\Isaac\IsaacLab-Arena
git submodule update --init --recursive
```

如果 GitHub 下载很慢，优先让 Codex 检查网络、代理、Git LFS、以及是否需要重试。OpenPI server 构建脚本后面会用 codeload tarball 下载，通常比完整 git clone 稳。

## 3. 创建 Windows Python 环境

本机使用 Python 3.11。示例使用 conda prefix 环境，方便固定在 `C:\Isaac\envs\arena-py311`。

```powershell
conda create -p C:\Isaac\envs\arena-py311 python=3.11 -y
conda activate C:\Isaac\envs\arena-py311

python -m pip install --upgrade pip setuptools wheel
python -m pip install "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install packaging==23.0 psutil==5.9.8 typing_extensions==4.12.2 numpy==1.26.0 av
```

安装 Isaac Lab 子模块：

```powershell
Set-Location C:\Isaac\IsaacLab-Arena\submodules\IsaacLab
.\isaaclab.bat --install
```

安装 IsaacLab-Arena：

```powershell
Set-Location C:\Isaac\IsaacLab-Arena
python -m pip install -e .
```

安装 Windows 客户端侧的 `openpi-client`，提交号来自仓库内 `OPENPI_COMMIT`：

```powershell
Set-Location C:\Isaac\IsaacLab-Arena
$openpiCommit = (Get-Content .\isaaclab_arena_openpi\docker\OPENPI_COMMIT).Trim()
python -m pip install --timeout 120 --retries 10 "openpi-client @ https://codeload.github.com/Physical-Intelligence/openpi/tar.gz/$openpiCommit#subdirectory=packages/openpi-client"
```

快速验证：

```powershell
C:\Isaac\envs\arena-py311\python.exe -c "import isaacsim, isaaclab, isaaclab_arena; print('imports ok')"
C:\Isaac\envs\arena-py311\python.exe -m pip list --format=freeze | Select-String -Pattern '^(isaacsim|isaaclab|isaaclab_arena|torch|numpy|openpi-client|packaging|psutil|typing_extensions)'
```

期望能看到类似：

```text
isaacsim==5.1.0.0
isaaclab==4.5.24
isaaclab_arena==1.0.0
torch==2.7.0+cu128
numpy==1.26.0
openpi-client==0.1.0
packaging==23.0
psutil==5.9.8
typing_extensions==4.12.2
```

## 4. Windows 兼容补丁

这些补丁是本机跑通 Windows 原生 Isaac Sim 时实际需要的。另一台机器可以先尝试运行；如果遇到同类错误，再应用对应补丁。

### 4.1 `h5py` DLL 路径

文件：`C:\Isaac\IsaacLab-Arena\isaaclab_arena\evaluation\policy_runner.py`

在 imports 附近加入：

```python
import importlib.util
from pathlib import Path
```

并在 IsaacLab-Arena imports 之前加入：

```python
_H5PY_DLL_DIR_HANDLE = None
if os.name == "nt":
    h5py_spec = importlib.util.find_spec("h5py")
    if h5py_spec is not None and h5py_spec.origin is not None:
        _H5PY_DLL_DIR_HANDLE = os.add_dll_directory(str(Path(h5py_spec.origin).resolve().parent))
```

目的：Windows 上 `h5py` 的 DLL 搜索路径可能不在默认路径里，提前注册其目录，避免 recorder/HDF5 相关 import 失败。

### 4.2 可选 embodiment 缺依赖时跳过

文件：`C:\Isaac\IsaacLab-Arena\isaaclab_arena\embodiments\__init__.py`

将直接 star import 改成可选导入：

```python
from importlib import import_module

_EMBODIMENT_MODULES = (
    ".agibot.agibot",
    ".droid.droid",
    ".franka.franka",
    ".g1.g1",
    ".galbot.galbot",
    ".gr1t2.gr1t2",
    ".kuka_allegro.kuka_allegro",
)


for module_name in _EMBODIMENT_MODULES:
    try:
        module = import_module(module_name, package=__name__)
    except ModuleNotFoundError as exc:
        print(f"[isaaclab-arena] Skipping optional embodiment {module_name}: missing {exc.name}")
        continue
    globals().update({name: value for name, value in vars(module).items() if not name.startswith("_")})
```

目的：OpenPI DROID demo 不需要所有 embodiment；例如 `pinocchio` 缺失不应阻塞 DROID 任务。

### 4.3 Isaac Lab USD 下载锁兼容 Windows

文件：`C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\source\isaaclab\isaaclab\sim\spawners\from_files\from_files.py`

Windows 没有 `fcntl`。将顶部 `import fcntl` 改成：

```python
try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None
    import msvcrt
```

加入锁 helper：

```python
def _lock_file(lock_fd):
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    else:
        lock_fd.write("0")
        lock_fd.flush()
        lock_fd.seek(0)
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)


def _unlock_file(lock_fd):
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    else:
        lock_fd.seek(0)
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
```

把锁文件打开和释放改为：

```python
lock_fd = open(lock_path, "a+")  # noqa: SIM115
_lock_file(lock_fd)
```

以及：

```python
_unlock_file(lock_fd)
```

目的：多进程下载 USD 资源时在 Windows 上可运行。

### 4.4 排除无关 RTX lidar/radar sensor 扩展

文件：`C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\apps\isaaclab.python.rendering.kit`

将：

```toml
app.extensions.excluded = ["omni.kit.pip_archive"]
```

改成：

```toml
app.extensions.excluded = ["omni.kit.pip_archive", "isaacsim.sensors.rtx", "omni.sensors.nv.lidar", "omni.sensors.nv.radar"]
```

目的：这个 OpenPI camera demo 只需要 RGB camera，不需要 RTX lidar/radar。Windows 上这些扩展可能因为 `generic_model_output` DLL 加载失败而报错，排除后可减少干扰。

### 4.5 PhysX simulation view warmup 兜底

文件：`C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\source\isaaclab_physx\isaaclab_physx\physics\physx_manager.py`

在 `_warmup_and_create_views` 中做三类调整：

- 如果 `_view` 是 `None`，即使 `_warmup_needed` 已经是 false，也允许重新创建 view。
- `create_simulation_view("warp")` 返回 `None` 时，依次尝试 `"torch"`、`"numpy"`。
- view 创建成功后清理旧的 callback exception：`cls._callback_exception = None`。

参考本机最终 diff：

```diff
-        if not cls._warmup_needed:
+        if not cls._warmup_needed and cls._view is not None:
             return
+        if cls._view is None:
+            cls._view_created = False

-        if is_gpu:
-            cls._physx_sim.attach_stage(stage_id)
-
-        cls._physx.force_load_physics_from_usd()
-        cls._physx.start_simulation()
-        cls._physx.update_simulation(cls.get_physics_dt(), 0.0)
-        cls._physx_sim.fetch_results()
-        cls._event_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
-        cls._warmup_needed = False
+        if cls._warmup_needed:
+            if is_gpu:
+                cls._physx_sim.attach_stage(stage_id)
+            cls._physx.force_load_physics_from_usd()
+            cls._physx.start_simulation()
+            cls._physx.update_simulation(cls.get_physics_dt(), 0.0)
+            cls._physx_sim.fetch_results()
+            cls._event_bus.dispatch_event(IsaacEvents.PHYSICS_WARMUP.value, payload={})
+            cls._warmup_needed = False

         cls._view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
         cls._view_warp = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
+        if cls._view is None:
+            for backend in ("torch", "numpy"):
+                logger.warning("Failed to create warp simulation view. Trying '%s' backend.", backend)
+                cls._view = omni.physics.tensors.create_simulation_view(backend, stage_id=stage_id)
+                cls._view_warp = omni.physics.tensors.create_simulation_view(backend, stage_id=stage_id)
+                if cls._view is not None:
+                    break

-        cls._view_created = True
+        cls._view_created = cls._view is not None
+        if cls._view is None:
+            raise RuntimeError("Failed to create a PhysX simulation view.")
+        cls._callback_exception = None
```

目的：Windows 原生 Isaac Sim + pip IsaacLab 组合下，PhysX view 初始化可能先失败后恢复。这个兜底让 demo 能完成 rollout。

## 5. 构建 OpenPI server Docker 镜像

OpenPI server 在 WSL2 Ubuntu 里构建和运行。进入 WSL：

```powershell
wsl -d Ubuntu
```

在 WSL 中：

```bash
cd /mnt/c/Isaac/IsaacLab-Arena/isaaclab_arena_openpi/docker
./run_openpi_server.sh -r
```

说明：

- 该脚本会根据 `OPENPI_COMMIT` 下载 OpenPI，并构建 `isaaclab_arena_openpi-server:latest`。
- 本机将 OpenPI 下载方式改成 `https://codeload.github.com/.../tar.gz/<commit>`，比 git clone 更稳。
- 本机将 CUDA base image 从 Docker Hub 改成 NGC：`nvcr.io/nvidia/cuda`。
- 本机将 Dockerfile 中 uv managed Python 改为容器内 apt 安装的 Python 3.11，避免 GitHub release asset 下载超时。
- 脚本没有 build-only 参数，构建完会直接启动 GPU OpenPI server。镜像构好后可以 `Ctrl+C` 结束，后续建议用下面的 CPU server PowerShell 脚本运行。

检查镜像：

```powershell
wsl -d Ubuntu -- docker images isaaclab_arena_openpi-server
```

## 6. 创建 OpenPI server 辅助脚本

为了避免 12GB 显存同时给 Isaac Sim 和 OpenPI GPU server 抢资源，本机默认让 OpenPI server 用 CPU 推理。CPU server 通常占用约 16-19GB WSL 内存，但基本不占显存；Isaac Sim 原生端仍使用 GPU 渲染/仿真。

创建 `C:\Isaac\start_openpi_server.ps1`：

```powershell
$ErrorActionPreference = "Stop"

$containerName = "isaaclab_arena_openpi_server"

function Test-OpenPiPort {
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $connect = $client.BeginConnect("127.0.0.1", 8000, $null, $null)
        $ready = $connect.AsyncWaitHandle.WaitOne(1000)
        if ($ready) {
            $client.EndConnect($connect)
            $client.Close()
            return $true
        }
        $client.Close()
    } catch {
        return $false
    }
    return $false
}

if (Test-OpenPiPort) {
    Write-Host "OpenPI server is already reachable at 127.0.0.1:8000."
    exit 0
}

$bashCommand = @'
mkdir -p /root/.cache/openpi
docker rm -f isaaclab_arena_openpi_server >/dev/null 2>&1 || true
docker run -d --rm --name isaaclab_arena_openpi_server --network=host \
  -v /root/.cache/openpi:/root/.cache/openpi \
  -e JAX_PLATFORMS=cpu \
  isaaclab_arena_openpi-server:latest \
  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
'@

wsl.exe -d Ubuntu -- bash -lc $bashCommand

for ($i = 0; $i -lt 240; $i += 5) {
    Start-Sleep -Seconds 5
    if (Test-OpenPiPort) {
        Write-Host "OpenPI server is reachable at 127.0.0.1:8000."
        exit 0
    }
    Write-Host "Waiting for OpenPI server... ${i}s"
}

wsl.exe -d Ubuntu -- bash -lc "docker logs --tail 80 $containerName 2>&1 || true"
Write-Error "OpenPI server did not become reachable. See Docker logs above."
exit 1
```

创建 `C:\Isaac\stop_openpi_server.ps1`：

```powershell
$ErrorActionPreference = "Continue"

$bashCommand = @'
docker ps --filter ancestor=isaaclab_arena_openpi-server:latest -q | xargs -r docker stop
'@

wsl.exe -d Ubuntu -- bash -lc $bashCommand
```

启动服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\start_openpi_server.ps1
```

检查状态：

```powershell
wsl -d Ubuntu -- docker stats --no-stream isaaclab_arena_openpi_server
```

停止服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\stop_openpi_server.ps1
```

## 7. 跑官方 3-episode demo

创建 `C:\Isaac\run_openpi_demo.ps1`：

```powershell
$ErrorActionPreference = "Continue"

$repo = "C:\Isaac\IsaacLab-Arena"
$py = "C:\Isaac\envs\arena-py311\python.exe"
$out = "C:\Isaac\logs\arena_openpi_demo.log"
$err = "C:\Isaac\logs\arena_openpi_demo.err.log"
$exitFile = "C:\Isaac\logs\arena_openpi_demo.exit"

New-Item -ItemType Directory -Force -Path "C:\Isaac\logs" | Out-Null
Remove-Item -LiteralPath $out, $err, $exitFile -ErrorAction SilentlyContinue

function Test-OpenPiPort {
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $connect = $client.BeginConnect("127.0.0.1", 8000, $null, $null)
        $ready = $connect.AsyncWaitHandle.WaitOne(1000)
        if ($ready) {
            $client.EndConnect($connect)
            $client.Close()
            return $true
        }
        $client.Close()
    } catch {
        return $false
    }
    return $false
}

if (-not (Test-OpenPiPort)) {
    & "C:\Isaac\start_openpi_server.ps1"
    if ($LASTEXITCODE -ne 0) {
        $LASTEXITCODE | Set-Content -LiteralPath $exitFile
        exit $LASTEXITCODE
    }
}

$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE = "1"

$argsList = @(
    "C:\Isaac\IsaacLab-Arena\isaaclab_arena\evaluation\policy_runner.py",
    "--viz", "kit",
    "--experience", "C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\apps\isaaclab.python.rendering.kit",
    "--policy_type", "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy",
    "--num_episodes", "3",
    "--enable_cameras", "--num_envs", "1",
    "--language_instruction", "Pick up the Rubik's cube and place it in the bowl.",
    "--remote_host", "127.0.0.1", "--remote_port", "8000",
    "pick_and_place_maple_table",
    "--embodiment", "droid_abs_joint_pos",
    "--pick_up_object", "rubiks_cube_hot3d_robolab",
    "--destination_location", "bowl_ycb_robolab",
    "--hdr", "home_office_robolab"
)

Set-Location $repo
& $py @argsList > $out 2> $err
$LASTEXITCODE | Set-Content -LiteralPath $exitFile
exit $LASTEXITCODE
```

运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\run_openpi_demo.ps1
```

查看结果：

```powershell
Get-Content C:\Isaac\logs\arena_openpi_demo.exit
Select-String -Path C:\Isaac\logs\arena_openpi_demo.log -Pattern "Metrics:" | Select-Object -Last 5
```

本机官方 demo 跑通结果：

```text
exit = 0
success_rate = 0.6666666666666666
object_moved_rate = 1.0
num_episodes = 3
```

## 8. 录制视频

录制视频时加入 `--camera_video --video_dir <dir>`。视频在 episode 结束后才会写入 mp4，运行中目录为空是正常的。

示例 1：Rubik's cube -> bowl

```powershell
New-Item -ItemType Directory -Force -Path C:\Isaac\videos\openpi_demo_camera | Out-Null
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\start_openpi_server.ps1

$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE = "1"
Set-Location C:\Isaac\IsaacLab-Arena

& C:\Isaac\envs\arena-py311\python.exe `
  C:\Isaac\IsaacLab-Arena\isaaclab_arena\evaluation\policy_runner.py `
  --viz kit `
  --experience C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\apps\isaaclab.python.rendering.kit `
  --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy `
  --num_episodes 1 `
  --camera_video `
  --video_dir C:\Isaac\videos\openpi_demo_camera `
  --enable_cameras `
  --num_envs 1 `
  --language_instruction "Pick up the Rubik's cube and place it in the bowl." `
  --remote_host 127.0.0.1 `
  --remote_port 8000 `
  pick_and_place_maple_table `
  --embodiment droid_abs_joint_pos `
  --pick_up_object rubiks_cube_hot3d_robolab `
  --destination_location bowl_ycb_robolab `
  --hdr home_office_robolab
```

示例 2：mustard bottle -> wooden bowl

```powershell
New-Item -ItemType Directory -Force -Path C:\Isaac\videos\openpi_mustard_wooden_bowl_camera | Out-Null
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\start_openpi_server.ps1

$env:OMNI_KIT_ACCEPT_EULA = "Y"
$env:ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE = "1"
Set-Location C:\Isaac\IsaacLab-Arena

& C:\Isaac\envs\arena-py311\python.exe `
  C:\Isaac\IsaacLab-Arena\isaaclab_arena\evaluation\policy_runner.py `
  --viz kit `
  --experience C:\Isaac\IsaacLab-Arena\submodules\IsaacLab\apps\isaaclab.python.rendering.kit `
  --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy `
  --num_episodes 1 `
  --camera_video `
  --video_dir C:\Isaac\videos\openpi_mustard_wooden_bowl_camera `
  --enable_cameras `
  --num_envs 1 `
  --language_instruction "Pick up the mustard bottle and place it in the wooden bowl." `
  --remote_host 127.0.0.1 `
  --remote_port 8000 `
  pick_and_place_maple_table `
  --embodiment droid_abs_joint_pos `
  --pick_up_object mustard_bottle_hot3d_robolab `
  --destination_location wooden_bowl_hot3d_robolab `
  --hdr home_office_robolab
```

成功后会生成三个相机视频：

```text
robot-cam-external_camera_rgb-step-0.mp4
robot-cam-external_camera_2_rgb-step-0.mp4
robot-cam-wrist_camera_rgb-step-0.mp4
```

本机 mustard bottle demo 结果：

```text
exit = 0
success_rate = 1.0
object_moved_rate = 1.0
num_episodes = 1
```

## 9. GPU/显存说明

本机默认方案：

- Isaac Sim：Windows 原生运行，使用 GPU 渲染/仿真。日志中能看到 `sim.device='cuda:0'`。
- OpenPI server：WSL Docker 中 CPU 推理，设置了 `JAX_PLATFORMS=cpu`。

为什么 OpenPI 默认不用 GPU：

- 在 RTX 4070 Ti 12GB 上，单独启动 OpenPI GPU server 会让 dedicated VRAM 从约 2.1GB 涨到约 11.2GB，OpenPI 自身约占 9.1GB dedicated VRAM。
- 任务管理器可能同时显示 shared GPU memory，例如总 GPU memory 到 13.3GB。这不等于 13.3GB 都是真实显存；dedicated GPU memory 才是板载 VRAM。
- 同时运行 Isaac Sim 和 OpenPI GPU server 很容易把 12GB VRAM 打满，所以本机稳定方案是 OpenPI CPU + Isaac Sim GPU。

如果另一台机器有 24GB 或更大显存，可以测试 OpenPI GPU server：

```powershell
wsl -d Ubuntu -- docker run -d --rm --name openpi_gpu_probe --gpus all --network=host `
  -v /root/.cache/openpi:/root/.cache/openpi `
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false `
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 `
  isaaclab_arena_openpi-server:latest `
  uv run scripts/serve_policy.py policy:checkpoint `
    --policy.config=pi05_droid_jointpos_polaris `
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

监控：

```powershell
nvidia-smi
wsl -d Ubuntu -- docker stats --no-stream openpi_gpu_probe
```

停止：

```powershell
wsl -d Ubuntu -- docker stop openpi_gpu_probe
```

## 10. 常见问题

### WSL2/Docker 里 Isaac Sim 为什么不可用

当前机器的 WSL2 环境没有暴露 NVIDIA Vulkan/RTX ICD，`vulkaninfo` 只能看到 llvmpipe CPU Vulkan。Isaac Sim 需要 Vulkan/RTX 渲染通路，不能靠 CPU Vulkan 正常跑 camera/RTX pipeline。因此使用 Windows 原生 Isaac Sim。

### 日志里的 lidar/radar/generic_model_output 报错

如果 demo 已经进入 rollout 并能输出 metrics，这些通常是非阻塞的扩展加载噪音。camera demo 不需要 lidar/radar。建议应用上面的 `isaaclab.python.rendering.kit` 排除补丁。

### 视频目录一直空

正常。`--camera_video` 的 mp4 通常在 episode 结束后才 flush。确认 rollout 是否在跑：

```powershell
Get-Process python -ErrorAction SilentlyContinue
Select-String -Path C:\Isaac\logs\*.log -Pattern "Starting rollout|Metrics|Closing simulation app"
```

### OpenPI server 启动慢

第一次启动要下载 checkpoint。观察 Docker logs：

```powershell
wsl -d Ubuntu -- docker logs -f isaaclab_arena_openpi_server
```

### 清理

停止 OpenPI：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Isaac\stop_openpi_server.ps1
```

如果需要强制清理容器：

```powershell
wsl -d Ubuntu -- docker rm -f isaaclab_arena_openpi_server
```

## 11. 给另一台机器 Codex 的建议工作流

1. 先验证 Windows `nvidia-smi`、WSL `nvidia-smi`、Docker GPU。
2. 克隆 IsaacLab-Arena 并初始化 submodules。
3. 在 Windows 原生 conda/venv 中安装 Isaac Sim、Isaac Lab、IsaacLab-Arena、openpi-client。
4. 应用 Windows 兼容补丁，尤其是 `h5py` DLL、`fcntl` fallback、sensor extension 排除。
5. 在 WSL Docker 中构建 `isaaclab_arena_openpi-server:latest`。
6. 用 CPU OpenPI server 跑官方 3-episode demo。
7. 再用 `--camera_video` 跑 1-episode 录制。
8. 只有在显存足够时再尝试 OpenPI GPU server。
