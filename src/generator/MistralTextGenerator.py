from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from .Generators import TextGenerator
from utils.logging import get_logger

logger = get_logger("MistralTextGenerator")

# Load environment variables (e.g., for consistency across generators, though not used directly here)
load_dotenv()

class MistralTextGenerator(TextGenerator):
    def __init__(self, model_name: str, generation_args: dict):
        super().__init__(model_name)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.generation_args = generation_args

        logger.debug(f"Initialized with model: {model_name}")
        logger.debug(f"[MistralTextGenerator] Model running on device: {self.device}")
        logger.debug(f"[MistralTextGenerator] Generation arguments: {self.generation_args}")

    def generate(self, prompt: str) -> str:
        logger.info(f"[MistralTextGenerator] Generating response using model: {self.model_name}")
        logger.debug(f"[MistralTextGenerator] Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self.generation_args
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[MistralTextGenerator] Response: {response}")
        return response