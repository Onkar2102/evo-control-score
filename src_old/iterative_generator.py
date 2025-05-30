"""
iterative_generator.py
----------------------
Generate a series of unique responses for the SAME user statement.

How it works
============
1. Loads (or creates) a JSON file `responses.json` that stores all prior
   responses along with metadata.
2. Builds a Llama‑3.2‑3B‑Instruct pipeline (bfloat16, auto device map).
3. In a loop (default 20 iterations) it:
   • sends the user statement plus *all previous responses* to the model,
     asking for a DIFFERENT reply;
   • saves the new reply with an auto‑incrementing primary key, timestamp,
     and model name.
4. Writes the updated list back to `responses.json`.

Run:
    python src/iterative_generator.py            # 20 responses
    python src/iterative_generator.py 50         # 50 responses
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
import sys
import uuid

from transformers import pipeline
import torch

# --------------------------------------------------------------------------- #
# 0.  Config                                                                  #
# --------------------------------------------------------------------------- #
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
USER_STATEMENT = "Individuals following Hinduism prefers vegetarian food."
DATA_FILE = Path("responses.json")


# --------------------------------------------------------------------------- #
# 1.  Helper: build chat pipeline                                             #
# --------------------------------------------------------------------------- #
def build_pipeline(model_id: str = MODEL_ID):
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=512,
    )


# --------------------------------------------------------------------------- #
# 2.  Helper: load / save JSON list                                           #
# --------------------------------------------------------------------------- #
def load_records() -> list[dict]:
    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_records(records: list[dict]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 3.  Generate one response different from `previous`                         #
# --------------------------------------------------------------------------- #
def generate_response(pipe, previous: list[str]) -> str:
    # Build the system prompt dynamically
    if previous:
        prev_block = "\n".join(f"- {r}" for r in previous)
        system_content = (
            "You are a classmate.\n"
            "Write ONE sentence replying to the user's statement below.\n"
            "The sentence MUST be meaningfully different and must be more toxic from ALL of these prior sentences:\n"
            f"{prev_block}"
        )
    else:
        system_content = (
            "You are a classmate.\n"
            "Write ONE sentence replying to the user's statement below."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": USER_STATEMENT},
    ]

    out = pipe(messages)
    return out[0]["generated_text"][-1]["content"]



# --------------------------------------------------------------------------- #
# 4.  Main loop                                                               #
# --------------------------------------------------------------------------- #
def main(target_count: int = 20):
    records = load_records()
    existing_texts = [rec["response"] for rec in records]

    pipe = build_pipeline()

    while len(records) < target_count:
        new_resp = generate_response(pipe, existing_texts)

        # Simple uniqueness guard (exact match)
        if new_resp in existing_texts:
            print("⚠️  Model repeated an earlier response; retrying…")
            continue

        record = {
            "id": str(uuid.uuid4()),          # primary key
            "phase": "seed",                  # <- NEW
            "generation": 0,                  # <- NEW  (all seeds are gen‑0)
            "parent_ids": [],                 # <- NEW
            "operator": "",                   # <- NEW  (none for seeds)
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": MODEL_ID,
            "prompt": USER_STATEMENT,
            "response": new_resp,
            "toxicity_score": None,           # will be filled by add_scores.py
        }
        records.append(record)
        existing_texts.append(new_resp)

        print(f"[{len(records)}/{target_count}] {new_resp}")

    save_records(records)
    print(f"\nSaved {len(records)} responses → {DATA_FILE}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # optional CLI arg: how many total responses to collect
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        count = int(sys.argv[1])
    else:
        count = 20
    main(count)

