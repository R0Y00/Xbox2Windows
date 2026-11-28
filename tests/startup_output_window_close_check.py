"""Minimize and close a source-built presenter in direct and managed hosts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    exe = Path(sys.argv[1]).resolve()
    debugger = (Path(sys.argv[2]) if len(sys.argv) > 2 else
                Path(__file__).resolve().parents[1] / "tools" / "startup" / "debug_launch.py")
    env = dict(os.environ, X2W_D3D12_RENDER="1", X2W_D3D12_PERMISSIVE="1",
               X2W_D3D12_LIST_NOOP="1", X2W_TEST_WINDOW_CLOSE="1")
    env.pop("X2W_SESSION_CLOSE_EVENT", None)
    with tempfile.TemporaryDirectory(prefix="x2w-window-close-") as directory:
        direct = subprocess.run([exe], cwd=directory, env=env, timeout=15, capture_output=True, text=True)
        if direct.returncode:
            print(direct.stdout + direct.stderr)
            return direct.returncode
        logs = Path(directory) / "managed"
        result = subprocess.run([sys.executable, "-B", debugger, exe, directory, logs,
                                 "--console", "0", "--managed-window", "1", "--timeout", "20"],
                                env=env, timeout=25, capture_output=True, text=True)
        if (result.returncode or "stopped by output window close" not in result.stdout or
                "exit 0x00000000" not in (logs / "events.txt").read_text(encoding="utf-8")):
            print(result.stdout + result.stderr)
            return 1
    print("Direct host survived Close, and managed Close ended the child and debugger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
