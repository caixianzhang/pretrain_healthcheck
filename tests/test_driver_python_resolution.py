from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
HELPER = PROJECT_DIR / "scripts" / "common" / "driver_python.sh"


class DriverPythonResolutionTest(unittest.TestCase):
    def run_resolver(self, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        command = (
            f'source "{HELPER}"; '
            'resolve_driver_python; rc=$?; '
            'printf "RC=%s\\nPY=%s\\nVERSION=%s\\n" '
            '"$rc" "${DRIVER_PYTHON:-}" "${DRIVER_PYTHON_VERSION:-}"; '
            'exit "$rc"'
        )
        return subprocess.run(
            ["/bin/bash", "-c", command],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in ("DRIVER_PYTHON", "DRIVER_PYTHON_VERSION", "DRIVER_PYTHON_RESOLVED", "CONDA_PREFIX"):
            env.pop(key, None)
        return env

    def test_explicit_supported_python(self) -> None:
        env = self.base_env()
        env["DRIVER_PYTHON"] = sys.executable
        proc = self.run_resolver(env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"PY={sys.executable}", proc.stdout)

    def test_explicit_missing_python_fails_without_fallback(self) -> None:
        env = self.base_env()
        env["DRIVER_PYTHON"] = "/missing/python3"
        proc = self.run_resolver(env)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("DRIVER_PYTHON=/missing/python3", proc.stderr)

    def test_explicit_python_38_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "python3"
            fake.write_text(
                "#!/bin/sh\n"
                "case \"$2\" in\n"
                "  *print*) echo 3.8.10; exit 0 ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = self.base_env()
            env["DRIVER_PYTHON"] = str(fake)
            proc = self.run_resolver(env)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("3.8.10", proc.stderr)

    def test_auto_falls_back_from_python_38_to_home_miniconda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            conda_bin = root / "home" / "miniconda3" / "bin"
            bin_dir.mkdir()
            conda_bin.mkdir(parents=True)
            fake = bin_dir / "python3"
            fake.write_text(
                "#!/bin/sh\n"
                "case \"$2\" in\n"
                "  *print*) echo 3.8.10; exit 0 ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            fallback = conda_bin / "python3"
            fallback.symlink_to(sys.executable)
            env = self.base_env()
            env["PATH"] = str(bin_dir)
            env["HOME"] = str(root / "home")
            proc = self.run_resolver(env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"PY={fallback}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
