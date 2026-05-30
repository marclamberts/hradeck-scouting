"""
Build a lightweight Wyscout player search index.

Reads every xlsx in data/Wyscout DB/ and extracts:
  Player, Team, Position, Age, File

Output: data/Wyscout_Player_Index.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

WYSCOUT_DIR = Path("data/Wyscout DB")
OUT_FILE = Path("data/Wyscout_Player_Index.csv")

USECOLS = {"Player", "Team", "Position", "Pos", "Age"}


def main() -> None:
    files = sorted(WYSCOUT_DIR.glob("*.xlsx"))
    print(f"Indexing {len(files)} Wyscout files…")

    chunks: list[pd.DataFrame] = []
    skipped = 0

    for xlsx in files:
        try:
            df = pd.read_excel(xlsx)
            # Normalise position column
            if "Position" not in df.columns and "Pos" in df.columns:
                df = df.rename(columns={"Pos": "Position"})
            keep = [c for c in ("Player", "Team", "Position", "Age") if c in df.columns]
            if "Player" not in keep:
                skipped += 1
                continue
            chunk = df[keep].copy()
            chunk["File"] = xlsx.name
            chunk = chunk.dropna(subset=["Player"])
            chunk = chunk[chunk["Player"].astype(str).str.strip() != ""]
            # Strip to first position (Wyscout stores comma-separated lists)
            if "Position" in chunk.columns:
                chunk["Position"] = (
                    chunk["Position"].astype(str).str.split(r"[,;]").str[0].str.strip()
                )
            chunks.append(chunk)
        except Exception as exc:
            print(f"  SKIP {xlsx.name}: {exc}")
            skipped += 1

    if not chunks:
        print("No data found.")
        return

    idx = pd.concat(chunks, ignore_index=True)
    idx.to_csv(OUT_FILE, index=False)
    print(f"\nIndex built: {len(idx):,} player rows from {len(files) - skipped} files")
    print(f"Saved → {OUT_FILE}  ({OUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
