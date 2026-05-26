#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pystamps.config import RunConfig
from pystamps.verify import comparison_failure_payload, verify_run_against_golden


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a narrow parity comparison against golden outputs.")
    parser.add_argument("--run", required=True, help="Run root to compare")
    parser.add_argument("--golden", required=True, help="Golden root to compare against")
    parser.add_argument("--patterns", nargs="+", required=True, help="One or more glob patterns to compare")
    parser.add_argument("--label", default="narrow_compare", help="Optional label for the comparison artifact")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = RunConfig()
    cfg.tolerance.rtol = args.rtol
    cfg.tolerance.atol = args.atol
    report = verify_run_against_golden(
        Path(args.run).expanduser().resolve(),
        Path(args.golden).expanduser().resolve(),
        cfg.tolerance,
        patterns=tuple(args.patterns),
    )
    failures = [comparison_failure_payload(c) for c in report.comparisons if not c.ok]
    payload = {
        "label": args.label,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(Path(args.run).expanduser().resolve()),
        "golden_root": str(Path(args.golden).expanduser().resolve()),
        "patterns": list(args.patterns),
        "ok": report.ok,
        "checked": len(report.comparisons),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
