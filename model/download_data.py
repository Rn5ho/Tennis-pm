"""Download Jeff Sackmann's tennis data repos."""

import subprocess
import sys
from pathlib import Path

from config.settings import SACKMANN_ATP_DIR, SACKMANN_WTA_DIR

REPOS = {
    "tennis_atp": ("https://github.com/JeffSackmann/tennis_atp.git", SACKMANN_ATP_DIR),
    "tennis_wta": ("https://github.com/JeffSackmann/tennis_wta.git", SACKMANN_WTA_DIR),
}


def download(force: bool = False) -> None:
    for name, (url, dest) in REPOS.items():
        dest = Path(dest)
        if dest.exists() and not force:
            print(f"{name}: already exists at {dest}, skipping (use --force to re-download)")
            continue

        if dest.exists() and force:
            import shutil
            shutil.rmtree(dest)

        print(f"{name}: cloning {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
        )
        print(f"{name}: done")

    # Verify key files exist
    for name, (_, dest) in REPOS.items():
        matches = list(Path(dest).glob("*_matches_????.csv"))
        print(f"{name}: found {len(matches)} match files")
        if not matches:
            print(f"  WARNING: no match files found in {dest}", file=sys.stderr)


if __name__ == "__main__":
    force = "--force" in sys.argv
    download(force=force)
