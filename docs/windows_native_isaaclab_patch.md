# Windows-native IsaacLab compatibility patch

This repository pins IsaacLab as a submodule. On the Windows-native Isaac Sim
setup used for local development, four compatibility changes are required but
must not be recorded as an unreachable commit inside the detached submodule.
They are therefore stored as `patches/isaaclab/windows-native.patch` in the
parent repository. The patch uses zero-context hunks so standard whitespace
hooks cannot corrupt unified-diff context markers.

The patch:

- excludes RTX sensor extensions unavailable in the local Isaac Sim runtime;
- uses `msvcrt` instead of `fcntl` for Windows file locking;
- makes PhysX simulation-view creation retry-safe; and
- falls back from Warp to Torch or NumPy PhysX tensor views.

Initialize IsaacLab and apply the patch from PowerShell:

```powershell
git submodule update --init -- submodules/IsaacLab
.\tools\apply_windows_native_isaaclab_patch.ps1
```

The script is idempotent. It verifies the pinned IsaacLab commit, reports an
already-applied patch without changing files, and refuses to run over unrelated
tracked or staged submodule changes.

The submodule will intentionally appear as `modified` after applying the patch.
Do not commit a new detached IsaacLab gitlink: such a commit would exist only on
the local workstation and could not be fetched by another clone. Commit updates
to the patch file in the Arena repository instead.
