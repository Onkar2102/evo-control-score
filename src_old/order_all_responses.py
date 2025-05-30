"""
order_all_responses.py
-----------------------
This script loads all responses from responses.json and orders
them by their prompt, then toxicity_score (ascending), and then by id.
If a record does not have a toxicity_score, it is assumed to be 1.0 (i.e. maximally toxic).

Usage:
    python order_all_responses.py
"""

import json
from pathlib import Path

DATA_FILE = Path("responses.json")

def load_records() -> list[dict]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found!")
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    records = load_records()

    # Sort by prompt (alphabetically), then by toxicity_score (lowest first, with missing scores as 1.0), then by id.
    sorted_records = sorted(
        records,
        key=lambda rec: (
            rec.get("prompt", ""),
            rec.get("toxicity_score") if rec.get("toxicity_score") is not None else 1.0,
            rec.get("id", "")
        )
    )
    
    print("Ordered responses:\n")
    for rec in sorted_records:
        print(f"ID: {rec.get('id')}")
        print(f"Prompt: {rec.get('prompt')}")
        print(f"Toxicity Score: {rec.get('toxicity_score') if rec.get('toxicity_score') is not None else 'N/A'}")
        print("Response:")
        print(rec.get("response"))
        print("-" * 50)

if __name__ == "__main__":
    main()