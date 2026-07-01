"""subprocess_runner.py — Python entry point for nightly cron re-indexing.

Called by reindex.sh. Runs train_books.py as a subprocess so that:
  - Any import errors or crashes are captured cleanly.
  - Exit code propagates correctly to cron/shell.
  - stdout and stderr are flushed in real time.
"""

import subprocess
import sys
from datetime import datetime


def main() -> int:
    print(f"{datetime.now().isoformat()} | subprocess_runner | Starting train_books.py ...")

    result = subprocess.run(
        [sys.executable, "train_books.py"],
        capture_output=False,  # Stream stdout/stderr directly (tee in shell handles logging)
        text=True,
    )

    if result.returncode == 0:
        print(f"{datetime.now().isoformat()} | subprocess_runner | Completed successfully.")
    else:
        print(
            f"{datetime.now().isoformat()} | subprocess_runner | "
            f"FAILED with exit code {result.returncode}.",
            file=sys.stderr,
        )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
