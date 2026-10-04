"""Detect the machine and size work to it: preview budget and memory checks."""
from __future__ import annotations

import os
import platform
import sys

import psutil

# Rough peak bytes per triangle, measured on the 15M-triangle benchmark.
BYTES_PER_FACE = {
    "load": 130,
    "analyze": 150,
    "repair": 260,
    "cut": 110,
    "preview": 60,
    "export": 40,
    "transform": 80,
    "boolean": 520,
}
SAFE_FRACTION = 0.8


def system_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count() or 1,
        "ram_total": vm.total,
        "ram_available": vm.available,
    }


def gpu_tier(renderer: str) -> str:
    r = (renderer or "").lower()
    if any(k in r for k in ("llvmpipe", "softpipe", "swiftshader", "software", "gdi generic")):
        return "software"
    if any(k in r for k in ("nvidia", "geforce", "quadro", "rtx", "radeon rx", "radeon pro")):
        return "dedicated"
    if "amd" in r or "radeon" in r:
        return "integrated"  # Radeon Vega/Graphics APUs; discrete ones match above
    return "integrated"


def display_budget(renderer: str, ram_total: int | None = None) -> int:
    """Max triangles drawn for all visible parts together."""
    tier = gpu_tier(renderer)
    budget = {"dedicated": 6_000_000, "integrated": 2_500_000, "software": 800_000}[tier]
    ram = ram_total or psutil.virtual_memory().total
    if ram < 12 * 2**30:
        budget = min(budget, 1_500_000)
    return budget


def estimate_bytes(op: str, n_faces: int) -> int:
    return BYTES_PER_FACE.get(op, 150) * max(n_faces, 0)


def check_memory(op: str, n_faces: int):
    """(ok, needed_bytes, available_bytes)."""
    need = estimate_bytes(op, n_faces)
    avail = psutil.virtual_memory().available
    return need <= SAFE_FRACTION * avail, need, avail


GPU_PREF_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"


def prefer_high_performance_gpu(exe_path: str) -> bool:
    """Windows: ask for the discrete GPU on dual-GPU laptops (from the next start).

    Writes the same per-user setting as Settings > Display > Graphics settings.
    Returns True when the preference was newly written.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, GPU_PREF_KEY) as key:
            try:
                value, _ = winreg.QueryValueEx(key, exe_path)
                if "GpuPreference=2" in value:
                    return False
            except FileNotFoundError:
                pass
            winreg.SetValueEx(key, exe_path, 0, winreg.REG_SZ, "GpuPreference=2;")
            return True
    except OSError:
        return False
