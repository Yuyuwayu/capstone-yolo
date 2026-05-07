"""
FishWatch — Environment Health Checker

Verifies that all required packages and directories are present.
Can auto-install missing dependencies via poetry/pip.
"""

import importlib
import subprocess
import sys
import os
import threading
from collections import deque

from . import config


class EnvChecker:
    def __init__(self):
        self.install_running = False
        self.install_log = deque(maxlen=300)
        self._install_thread = None

    # ── Checks ────────────────────────────────────────

    def check_all(self):
        """Full environment health check."""
        packages = self._check_packages()
        python = self._check_python()
        directories = self._check_directories()

        all_ok = (
            all(p["installed"] for p in packages)
            and python["ok"]
            and all(d["exists"] for d in directories)
        )

        return {
            "packages": packages,
            "python": python,
            "directories": directories,
            "all_ok": all_ok,
        }

    def _check_packages(self):
        results = []
        for import_name, pip_name in config.REQUIRED_PACKAGES.items():
            try:
                mod = importlib.import_module(import_name)
                version = getattr(mod, "__version__", "installed")
                results.append({"name": pip_name, "installed": True, "version": version})
            except ImportError:
                results.append({"name": pip_name, "installed": False, "version": None})
        return results

    def _check_python(self):
        v = sys.version_info
        version = f"{v.major}.{v.minor}.{v.micro}"
        ok = v.major == 3 and v.minor >= 13
        return {"version": version, "ok": ok}

    def _check_directories(self):
        results = []
        for dir_path in config.REQUIRED_DIRECTORIES:
            exists = os.path.isdir(dir_path)
            empty = not os.listdir(dir_path) if exists else True
            results.append({
                "path": os.path.relpath(dir_path, config.BASE_DIR),
                "exists": exists,
                "empty": empty,
            })
        return results

    # ── Install ───────────────────────────────────────

    def install_missing(self):
        """Kick off background installation of missing packages."""
        if self.install_running:
            return {"status": "already_running"}

        self.install_running = True
        self.install_log.clear()
        self._install_thread = threading.Thread(target=self._run_install, daemon=True)
        self._install_thread.start()
        return {"status": "started"}

    def _run_install(self):
        try:
            self.install_log.append("[INFO] Running poetry install ...")
            proc = subprocess.Popen(
                [sys.executable, "-m", "poetry", "install"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=config.BASE_DIR,
            )
            for line in iter(proc.stdout.readline, ""):
                if line:
                    self.install_log.append(line.rstrip())
            proc.wait()

            if proc.returncode == 0:
                self.install_log.append("[OK] poetry install succeeded.")
            else:
                self.install_log.append(f"[WARN] poetry exited with code {proc.returncode}, trying pip ...")
                self._pip_fallback()
        except FileNotFoundError:
            self.install_log.append("[WARN] poetry not found, falling back to pip ...")
            self._pip_fallback()
        except Exception as e:
            self.install_log.append(f"[ERROR] {e}")
        finally:
            self.install_running = False

    def _pip_fallback(self):
        missing = [p["name"] for p in self._check_packages() if not p["installed"]]
        if not missing:
            self.install_log.append("[OK] No missing packages.")
            return
        self.install_log.append(f"[INFO] pip install {' '.join(missing)}")
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in iter(proc.stdout.readline, ""):
            if line:
                self.install_log.append(line.rstrip())
        proc.wait()
        if proc.returncode == 0:
            self.install_log.append("[OK] pip install succeeded.")
        else:
            self.install_log.append(f"[ERROR] pip install failed (code {proc.returncode}).")

    def get_install_status(self):
        return {"running": self.install_running, "logs": list(self.install_log)}
