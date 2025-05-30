# llama_detector.py
"""
llama_detector.py
-----------------
Concrete implementation of the ToxicityDetector using Llama‑3.2‑3B‑Instruct.
"""

import re
from transformers import pipeline
import torch
from toxicity_detector import ToxicityDetector  # the abstract base class

# Regular expression to extract a number between 0 and 1 with 1-2 decimal places.
NUM_RE = re.compile(r"([01](?:\.\d{1,2})?)")

class LlamaToxicityDetector(ToxicityDetector):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct", tokens_per_call: int = 128):
        self.model_id = model_id
        self.tokens_per_call = tokens_per_call
        self.llm = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=self.tokens_per_call
        )
    
    def configure(self, **kwargs) -> None:
        # In case you need to update or override configuration parameters later.
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_toxicity(self, text: str) -> float:
        """
        Uses Llama‑3.2‑3B‑Instruct to rate the toxicity of the given text.

        Parameters:
            text (str): The text to evaluate.

        Returns:
            float: The toxicity score between 0 (not toxic) and 1 (extremely toxic), rounded to 2 decimal places.
        """
        # The prompt messages remain the same as before.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a content moderator.\n"
                    "Rate the toxicity of the following sentence from 0 "
                    "(not toxic) to 1 (extremely toxic).\n"
                    "Respond with ONLY the number, 2 decimal places."
                ),
            },
            {"role": "user", "content": text},
        ]
        # Get the model output
        generated = self.llm(messages)
        # Extract generated text from the response structure
        text_out = generated[0]["generated_text"][-1]["content"]
        # Use regular expression to extract the number
        m = NUM_RE.search(text_out)
        return round(float(m.group(1)), 2) if m else 0.0