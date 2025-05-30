"""
add_scores.py
-------------
Update responses.json IN‑PLACE by adding a `toxicity_score` field
for every record using the LlamaToxicityDetector.
"""

from pathlib import Path
import json
from typing import List, Dict

from llama_detector import LlamaToxicityDetector  # Import the concrete detector

DATA_FILE = Path("responses.json")

# ------------------------------------------------------------------ #
# 1.  Load existing records                                          #
# ------------------------------------------------------------------ #
if not DATA_FILE.exists():
    raise FileNotFoundError("responses.json not found – run iterative_generator first.")

with DATA_FILE.open("r", encoding="utf-8") as f:
    records: List[Dict] = json.load(f)

# ------------------------------------------------------------------ #
# 2.  Instantiate the Llama toxicity detector                          #
# ------------------------------------------------------------------ #
detector = LlamaToxicityDetector()

# ------------------------------------------------------------------ #
# 3.  Iterate over records and update toxicity_score                   #
# ------------------------------------------------------------------ #
updated = 0
for rec in records:
    # You can uncomment the next lines if you want to skip records that are already scored.
    # if "toxicity_score" in rec:
    #     continue  # already scored
    
    score = detector.get_toxicity(rec["response"])
    rec["toxicity_score"] = score
    updated += 1
    print(f"Scored → {score:.2f} | {rec['response'][:60]}…")

# ------------------------------------------------------------------ #
# 4.  Write updated records back to the same file                      #
# ------------------------------------------------------------------ #
if updated:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nAdded scores for {updated} records – saved to {DATA_FILE}")
else:
    print("All records already had toxicity_score – nothing changed.")