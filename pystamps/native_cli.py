from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pystamps.native import native_binary_command


def main() -> int:
    command, cwd = native_binary_command(skip_path=sys.argv[0])
    command.extend(sys.argv[1:])
    if hasattr(os, "execv") and cwd is None:
        os.execv(command[0], command)
    completed = subprocess.run(command, cwd=Path(cwd) if cwd else None, check=False)
    return int(completed.returncode)
