"""Drop all ORM tables and recreate empty schema (uses DATABASE_URL from .env)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database.session import close_db, reset_db  # noqa: E402


async def _main() -> None:
    await reset_db()
    await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required confirmation; without this flag the script exits without changes",
    )
    args = parser.parse_args()
    if not args.yes:
        print(
            "Refusing to run without --yes (this deletes all data in the configured database).",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(_main())
    print("Done: schema recreated; all previous rows removed.")


if __name__ == "__main__":
    main()
