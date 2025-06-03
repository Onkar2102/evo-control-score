import os
import json
import torch
import yaml
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logging import get_logger

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

class LlaMaTextGenerator:
    def __init__(self, model_key="llama", config_path="config/modelConfig.yaml", log_file=None):
        self.logger = get_logger("LLaMaTextGenerator", log_file)

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        self.logger.info(f"Loading LLaMA model: {self.model_cfg['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["name"], legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_cfg["name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        self.generation_args = self.model_cfg.get("generation_args", {})

        # Prompt template support
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_format = tmpl.get("format", "{{prompt}}")
        self.user_prefix = tmpl.get("user_prefix", "")
        self.assistant_prefix = tmpl.get("assistant_prefix", "")

    def format_prompt(self, raw_prompt):
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response(self, prompt):
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_args)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # return decoded, formatted
        llm_reply = decoded.replace(formatted + "\n", "").strip()
        return llm_reply.split('Adult 2:')[-1].strip()

    def process_population(self, pop_path="outputs/Population.json"):
        with open(pop_path, "r") as f:
            population = json.load(f)

        updated = False
        for genome in population:
            if genome.get("status") == "pending_generation":
                prompt = genome.get("prompt", "")
                self.logger.debug(f"Generating for genome ID {genome['id']} | prompt_id {genome['prompt_id']}")
                response = self.generate_response(prompt)
                genome["generated_response"] = response
                genome["status"] = "pending_evaluation"
                genome["model_provider"] = self.model_cfg.get("provider", "")
                genome["model_name"] = self.model_cfg.get("name", "")
                updated = True

        if updated:
            with open(pop_path, "w") as f:
                json.dump(population, f, indent=2)
            self.logger.info(f"Updated population saved to {pop_path}")
        else:
            self.logger.info("No genomes required generation.")
