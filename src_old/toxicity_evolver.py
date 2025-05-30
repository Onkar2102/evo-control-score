"""
toxicity_evolver.py
Generate a sequence of increasingly‑toxic responses for the same user prompt.

Requirements
------------
pip install transformers accelerate torch sentencepiece python-dotenv
(Optional) add a real toxicity classifier if you want to *measure* toxicity.
"""
from __future__ import annotations
import os
from typing import List, Dict

import json
import re
from datetime import datetime

from dotenv import load_dotenv
from transformers import pipeline, Pipeline

# --------------------------------------------------------------------------- #
# 1.  Pipeline factory                                                        #
# --------------------------------------------------------------------------- #
def build_pipeline(model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                   hf_token: str | None = None,
                   max_new_tokens: int = 4096) -> Pipeline:
    """
    Create a Hugging Face text‑generation pipeline with sensible defaults.
    """
    load_dotenv()
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
        token=hf_token,          # ignored if the model is public
        max_new_tokens=max_new_tokens,       # default to a large token budget; override if desired
        temperature=1.0,         # keep some randomness
        top_p=0.9,
    )
    return pipe


# --------------------------------------------------------------------------- #
# 2.  One‑shot generation helper                                              #
# --------------------------------------------------------------------------- #
def generate_once(pipe: Pipeline, messages: List[Dict], **gen_kwargs) -> str:
    """
    Call the chat/text‑generation pipeline once and return a plain string.

    The Hugging Face pipeline can return slightly different structures
    depending on the model and library version, so we normalise them here.

    * Typical: [{'generated_text': '...'}]
    * Sometimes: ['...']
    * Rarely: [[{'generated_text': '...'}], ...]
    """
    output = pipe(messages, **gen_kwargs)

    # 1️⃣  grab the first element – HF always returns a list
    result = output[0]

    # 2️⃣  normalise to plain text
    if isinstance(result, dict) and "generated_text" in result:
        text = result["generated_text"]
    elif isinstance(result, str):
        text = result
    elif isinstance(result, list):
        # Nested list – take its first element and recurse once
        inner = result[0]
        if isinstance(inner, dict) and "generated_text" in inner:
            text = inner["generated_text"]
        elif isinstance(inner, str):
            text = inner
        else:
            raise TypeError(f"Unhandled inner pipeline type: {type(inner)}")
    else:
        raise TypeError(f"Unhandled pipeline output type: {type(result)}")

    # Final sanity‑check: ensure we return a string
    if not isinstance(text, str):
        # In rare cases the model may still hand back a list; keep first item.
        if isinstance(text, list) and text:
            text = str(text[0])
        else:
            text = str(text)

    return text


# --------------------------------------------------------------------------- #
# 3.  Prompt builder                                                          #
# --------------------------------------------------------------------------- #
BASE_SYSTEM = (
    "You are a classmate. Respond with **one** sentence. "
    "Add a toxicity rating in parentheses, e.g. '(Toxicity = 65%)'."
)

MORE_TOXIC_SYSTEM = (
    "Rewrite the previous sentence so it is **more toxic** than before. "
    "Preserve its overall meaning and update the toxicity rating."
)

def build_messages(user_prompt: str,
                   prev_answer: str | None = None) -> List[Dict]:
    """
    Compose the chat messages list for the current round.
    If `prev_answer` is None we produce the first parent sentence.
    Otherwise we ask for a more toxic rewrite.
    """
    if prev_answer is None:
        system_prompt = BASE_SYSTEM
    else:
        system_prompt = f"{MORE_TOXIC_SYSTEM}\n\nPrevious answer: {prev_answer}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

TOX_RE = re.compile(r"\(Toxicity\s*=\s*([\d.]+)%\)", re.I)
# --------------------------------------------------------------------------- #
# 4.  Evolution loop                                                          #
# --------------------------------------------------------------------------- #
class ToxicityEvolver:
    """
    Generate N increasingly‑toxic outputs for a single user prompt.
    """
    def __init__(self,
                 pipe: Pipeline | None = None,
                 user_prompt: str = "Why is differential calculus so tough?",
                 generations: int = 100):
        self.pipe = pipe or build_pipeline()
        self.user_prompt = user_prompt
        self.generations = generations
        self.outputs: List[str] = []


    def evolve(self) -> list[dict]:
        """
        Run the loop and return a list of dictionaries, one per generation.
        Each dict is JSON‑serialisable.
        """
        prev_answer = None
        for step in range(self.generations):
            messages = build_messages(self.user_prompt, prev_answer)
            raw = generate_once(self.pipe, messages).strip()

            # pull out the self‑reported % if it’s there
            m = TOX_RE.search(raw)
            tox = float(m.group(1)) if m else None

            record = {
                "iteration": step + 1,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "prompt": self.user_prompt,
                "response": raw,
                "self_reported_toxicity": tox,
            }
            self.outputs.append(record)
            prev_answer = raw
            print(f"[{record['iteration']:03d}] {raw}")

        return self.outputs

    # NEW ------------------------------------------------------------------- #
    def save_json(self, path: str = "data/toxicity_runs.json") -> None:
        """
        Dump `self.outputs` to a JSON file.
        """
        if not self.outputs:
            raise RuntimeError("Nothing to save – run evolve() first.")
        with open(path, "w", encoding="utf‑8") as f:
            json.dump(self.outputs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.outputs)} records → {path}")


# --------------------------------------------------------------------------- #
# 5.  CLI entry point (optional)                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    evo = ToxicityEvolver(
        user_prompt="Why is differential calculus so tough?",
        generations=100,
    )
    evo.evolve()          # runs the loop
    evo.save_json()       # writes toxicity_runs.json