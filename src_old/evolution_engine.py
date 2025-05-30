"""
evolution_engine.py
-------------------
Simple evolutionary optimisation over the 20 responses stored in responses.json.

* Objective: maximise (or minimise) `toxicity_score`
* Variation: random operator chosen from text_variation_operators.OPERATORS
* Selection: tournament of size 3
* Population size stays constant (= 20)

Usage
-----
    python src/evolution_engine.py 50        # 50 generations
"""

from __future__ import annotations
import random
import sys
import json
from pathlib import Path
from datetime import datetime
import uuid
from copy import deepcopy

from transformers import pipeline
import torch

from text_variation_operators import OPERATORS

DATA_FILE = Path("responses.json")
MODEL_ID   = "meta-llama/Llama-3.2-3B-Instruct"
POP_SIZE   = 20
TOUR_SIZE  = 3
MAX_NEW_TOK = 64   # tiny budget for scoring prompt


# --------------------------------------------------------------------------- #
# 1.  Utility: load / save                                                    #
# --------------------------------------------------------------------------- #
def load_population() -> list[dict]:
    if not DATA_FILE.exists():
        raise FileNotFoundError("responses.json not found. Run iterative_generator first.")
    with DATA_FILE.open("r", encoding="utf-8") as f:
        pop = json.load(f)
    if len(pop) < POP_SIZE:
        raise ValueError(f"Need at least {POP_SIZE} individuals in responses.json.")
    return pop[:POP_SIZE]  # first 20


def save_population(pop: list[dict]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(pop, f, ensure_ascii=False, indent=2)
        

from add_scores import build_scorer, get_toxicity_score   # NEW import

scorer = build_scorer(max_new_tokens=MAX_NEW_TOK)

def score_toxicity(text: str) -> float:
    return get_toxicity_score(text, llm=scorer)   # pass the shared pipeline


# --------------------------------------------------------------------------- #
# 3.  Evolutionary operators                                                  #
# --------------------------------------------------------------------------- #
def tournament_select(pop: list[dict], k: int = TOUR_SIZE) -> dict:
    return min(random.sample(pop, k), key=lambda ind: ind["toxicity_score"])


def evolve(generations: int = 50):
    pop = load_population()

    for gen in range(1, generations + 1):
        parent1 = tournament_select(pop)
        operator = random.choice(OPERATORS)

        if operator.arity == 1:
            child_text = operator.apply([parent1["response"]])
        else:
            parent2 = tournament_select(pop)
            child_text = operator.apply([parent1["response"], parent2["response"]])

        child_score = score_toxicity(child_text)

        # Replace worst individual if child is better (lower toxicity)
        worst = max(pop, key=lambda ind: ind["toxicity_score"])
        if child_score < worst["toxicity_score"]:
            child = {
                    "id": str(uuid.uuid4()),
                    "phase": "evo",
                    "generation": parent1.get("generation", 0) + 1,
                    "parent_ids": [parent1["id"]] if operator.arity == 1
                                else [parent1["id"], parent2["id"]],
                    "operator": operator.name,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "model": MODEL_ID,
                    "prompt": parent1["prompt"],
                    "response": child_text,
                    "toxicity_score": child_score,
            }
            print(f"[Gen {gen}] added child score {child_score:.4f} via {operator.name}")
            pop.append(child)

    save_population(pop)
    print("\nEvolution complete – population saved back to responses.json")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    gens = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 50
    evolve(gens)