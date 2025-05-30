"""
reset_scores.py
---------------
Set every `toxicity_score` in responses.json to null (None).

Usage
-----
    python src/reset_scores.py
"""

from __future__ import annotations
from pathlib import Path
import json

DATA_FILE = Path("responses.json")

if not DATA_FILE.exists():
    raise FileNotFoundError("responses.json not found.")

with DATA_FILE.open("r", encoding="utf-8") as f:
    records = json.load(f)

for rec in records:
    rec["toxicity_score"] = None

with DATA_FILE.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Reset toxicity_score for {len(records)} records → {DATA_FILE}")