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
    _MODEL_CACHE = {}
    def __init__(self, model_key="llama", config_path="config/modelConfig.yaml", log_file: str = None):
        print(log_file)
        self.log_file = log_file
        print(self.log_file)
        self.logger = get_logger("LLaMaTextGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if model_key not in config:
            raise ValueError(f"Model '{model_key}' not found in configuration.")
        self.model_cfg = config[model_key]

        model_name = self.model_cfg["name"]
        if model_name not in self._MODEL_CACHE:
            self.logger.info(f"Loading LLaMA model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.logger.info(f"Model loaded on device: {device}")
            model.to(device)
            model.eval()
            self._MODEL_CACHE[model_name] = (tokenizer, model)
        else:
            self.logger.info(f"Using cached LLaMA model: {model_name}")
        self.tokenizer, self.model = self._MODEL_CACHE[model_name]

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

    def generate_for_genome(self, genome):
        """
        Generate a response for a single genome dictionary in-place.
        """
        prompt = genome.get("prompt", "")
        self.logger.debug(f"Generating for genome ID {genome['id']} | prompt_id {genome['prompt_id']}")
        try:
            response = self.generate_response(prompt)
            genome["generated_response"] = response
            genome["status"] = "pending_evaluation"
            genome["model_provider"] = self.model_cfg.get("provider", "")
            genome["model_name"] = self.model_cfg.get("name", "")
        except Exception as e:
            self.logger.error(f"Failed to generate for genome ID {genome['id']}: {e}")
            raise e

    def convert_text_to_tokens(self, text):
        """
        Convert input text to its token IDs using the model's tokenizer.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0].tolist()
        return input_ids

    def convert_population_texts_to_tokens(self, pop_path="outputs/Population.json"):
        """
        Convert prompts and generated responses to token IDs for genomes
        with status 'pending_evolution' or 'most_toxic'.
        """
        self.logger.info("Converting prompt and generated response to token IDs...")

        try:
            with open(pop_path, "r") as f:
                population = json.load(f)

            updated = False
            for genome in population:
                if genome.get("status") in ["pending_evolution", "most_toxic"]:
                    prompt = genome.get("prompt", "")
                    response = genome.get("generated_response", "")
                    if prompt and response:
                        genome["input_tokens"] = self.convert_text_to_tokens(prompt)
                        genome["output_tokens"] = self.convert_text_to_tokens(response)
                        updated = True

            if updated:
                with open(pop_path, "w") as f:
                    json.dump(population, f, indent=2)
                self.logger.info("Updated population with input/output tokens.")
            else:
                self.logger.info("No genomes required token conversion.")
        except Exception as e:
            self.logger.error(f"Failed to convert texts to tokens: {e}")



    def paraphrase_text(self, text: str, num_variants: int = 2) -> list:
        """
        Generate at most `num_variants` paraphrased versions of the input text.
        """
        self.logger.info(f"Generating up to {num_variants} paraphrased variants for input text.")

        instruction = f"Paraphrase the following statement in different ways, keeping the meaning intact:\n{text}"
        self.logger.debug(f"Recieved Instruction - {instruction}")
        paraphrases = set()

        for i in range(num_variants * 2):  # Extra loops to ensure we get enough unique outputs
            formatted_prompt = self.format_prompt(instruction)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_args)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            generated = decoded.replace(formatted_prompt + "\n", "").strip()
            if generated.lower() != text.lower():
                paraphrases.add(generated)
            if len(paraphrases) >= num_variants:
                break

        paraphrase_list = list(paraphrases)
        self.logger.info(f"Generated {len(paraphrase_list)} unique paraphrased variants.")
        return paraphrase_list if paraphrase_list else [text]