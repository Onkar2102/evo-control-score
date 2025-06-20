import os
import json
import torch
import yaml
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.custom_logging import get_logger
from typing import List, Dict
import time

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

class LlaMaTextGenerator:
    _MODEL_CACHE = {}
    _DEVICE_CACHE = None
    
    def __init__(self, model_key="llama", config_path="config/modelConfig.yaml", log_file: str = None):
        self.log_file = log_file
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
            self._load_model_optimized(model_name)
        else:
            self.logger.info(f"Using cached LLaMA model: {model_name}")

        self.tokenizer, self.model, self.device = self._MODEL_CACHE[model_name]
        self.generation_args = self.model_cfg.get("generation_args", {})

        # Prompt template support
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_format = tmpl.get("format", "{{prompt}}")
        self.user_prefix = tmpl.get("user_prefix", "")
        self.assistant_prefix = tmpl.get("assistant_prefix", "")

        # Optimization settings
        self.max_batch_size = self.model_cfg.get("max_batch_size", 4)
        self.logger.info(f"Model loaded on {self.device} with batch size {self.max_batch_size}")

    def _get_optimal_device(self):
        """Get the best available device for M3 Mac"""
        if self._DEVICE_CACHE is not None:
            return self._DEVICE_CACHE
            
        if torch.backends.mps.is_available():
            self._DEVICE_CACHE = "mps"
            self.logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
        elif torch.cuda.is_available():
            self._DEVICE_CACHE = "cuda"
            self.logger.info("Using CUDA")
        else:
            self._DEVICE_CACHE = "cpu"
            self.logger.info("Using CPU")
        return self._DEVICE_CACHE

    def _load_model_optimized(self, model_name: str):
        """Load model with M3 Mac optimizations"""
        device = self._get_optimal_device()
        
        # Load tokenizer with optimizations
        self.logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            legacy=False,
            use_fast=True,  # Use fast tokenizer for better performance
            padding_side="left"  # Better for batch generation
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure model loading for M3 Mac
        model_kwargs = {
            "torch_dtype": torch.float16,  # Use half precision for memory efficiency
            "low_cpu_mem_usage": True,
            "device_map": None  # We'll manually move to device
        }
        
        # Add quantization for memory efficiency on M3
        if device == "mps":
            # MPS doesn't support all quantization yet, use float16
            model_kwargs["torch_dtype"] = torch.float16
            self.logger.info("Using float16 for MPS optimization")
        elif device == "cuda":
            # Use 4-bit quantization if available
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                self.logger.info("Using 4-bit quantization for memory efficiency")
            except Exception as e:
                self.logger.warning(f"Quantization not available: {e}")
        
        self.logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device and optimize
        if device != "cpu":
            model = model.to(device)
        model.eval()
        
        # Enable optimizations
        if hasattr(torch.backends, 'mps') and device == "mps":
            # MPS specific optimizations
            torch.backends.mps.allow_tf32 = True
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info("Model compiled for faster inference")
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")
        
        self._MODEL_CACHE[model_name] = (tokenizer, model, device)

    def format_prompt(self, raw_prompt: str) -> str:
        return (
            self.prompt_format
            .replace("{{user_prefix}}", self.user_prefix)
            .replace("{{assistant_prefix}}", self.assistant_prefix)
            .replace("{{prompt}}", raw_prompt)
        )

    def generate_response_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts in a single batch"""
        if not prompts:
            return []
        
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        # Tokenize with padding for batch processing
        inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048  # Reasonable limit for prompts
        ).to(self.device)
        
        # Generate with optimized settings
        generation_kwargs = {
            **self.generation_args,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,  # Enable KV cache for efficiency
            "do_sample": self.generation_args.get("do_sample", False)
        }
        
        with torch.no_grad():
            with torch.autocast(device_type="mps" if self.device == "mps" else "cuda", enabled=True):
                outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            # Extract only the generated part
            formatted_prompt = formatted_prompts[i]
            if decoded.startswith(formatted_prompt):
                response = decoded[len(formatted_prompt):].strip()
            else:
                response = decoded
            
            # Clean up response
            if 'Adult 2:' in response:
                response = response.split('Adult 2:')[-1].strip()
            
            responses.append(response)
        
        return responses

    def generate_response(self, prompt: str) -> str:
        """Single prompt generation (backwards compatibility)"""
        return self.generate_response_batch([prompt])[0]

    def process_population(self, pop_path="outputs/Population.json"):
        """Process population with batch optimization"""
        start_time = time.time()
        
        with open(pop_path, "r") as f:
            population = json.load(f)

        # Collect pending genomes for batch processing
        pending_genomes = []
        pending_indices = []
        
        for idx, genome in enumerate(population):
            if genome.get("status") == "pending_generation":
                pending_genomes.append(genome)
                pending_indices.append(idx)

        if not pending_genomes:
            self.logger.info("No genomes require generation.")
            return

        self.logger.info(f"Processing {len(pending_genomes)} genomes in batches of {self.max_batch_size}")
        
        # Process in batches
        updated_count = 0
        for i in range(0, len(pending_genomes), self.max_batch_size):
            batch_genomes = pending_genomes[i:i + self.max_batch_size]
            batch_indices = pending_indices[i:i + self.max_batch_size]
            
            # Extract prompts for batch processing
            batch_prompts = [genome.get("prompt", "") for genome in batch_genomes]
            
            self.logger.debug(f"Generating batch {i//self.max_batch_size + 1} with {len(batch_prompts)} prompts")
            
            try:
                # Generate responses in batch
                batch_responses = self.generate_response_batch(batch_prompts)
                
                # Update genomes with responses
                for j, (genome_idx, response) in enumerate(zip(batch_indices, batch_responses)):
                    population[genome_idx]["generated_response"] = response
                    population[genome_idx]["status"] = "pending_evaluation"
                    population[genome_idx]["model_provider"] = self.model_cfg.get("provider", "")
                    population[genome_idx]["model_name"] = self.model_cfg.get("name", "")
                    updated_count += 1
                    
                    self.logger.debug(f"Generated for genome ID {population[genome_idx]['id']}")
                
                # Periodic save to avoid losing progress
                if (i + self.max_batch_size) % (self.max_batch_size * 4) == 0:
                    with open(pop_path, "w") as f:
                        json.dump(population, f, indent=2)
                    self.logger.debug(f"Intermediate save completed after {i + self.max_batch_size} genomes")
                    
            except Exception as e:
                self.logger.error(f"Batch generation failed for batch starting at {i}: {e}")
                # Continue with next batch
                continue

        # Final save
            with open(pop_path, "w") as f:
                json.dump(population, f, indent=2)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Updated {updated_count} genomes in {elapsed_time:.2f} seconds ({updated_count/elapsed_time:.2f} genomes/sec)")

    def generate_for_genome(self, genome: Dict):
        """Generate a response for a single genome dictionary in-place."""
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

    def convert_text_to_tokens(self, text: str) -> List[int]:
        """Convert input text to its token IDs using the model's tokenizer."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"][0].tolist()
        return input_ids

    def convert_population_texts_to_tokens(self, pop_path="outputs/Population.json"):
        """Convert prompts and generated responses to token IDs for genomes
        with status 'pending_evolution' or 'most_toxic'."""
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

    def paraphrase_text(self, text: str, num_variants: int = 2) -> List[str]:
        """Generate at most `num_variants` paraphrased versions of the input text."""
        self.logger.info(f"Generating up to {num_variants} paraphrased variants for input text.")

        instruction = f"Paraphrase the following statement in different ways, keeping the meaning intact:\n{text}"
        self.logger.debug(f"Received Instruction - {instruction}")
        
        # Use batch generation for efficiency
        instructions = [instruction] * (num_variants * 2)  # Generate extra for uniqueness
        paraphrases = set()

        try:
            generated = self.generate_response_batch(instructions)
            for response in generated:
                if response.lower() != text.lower() and response.strip():
                    paraphrases.add(response.strip())
            if len(paraphrases) >= num_variants:
                break
        except Exception as e:
            self.logger.error(f"Batch paraphrasing failed: {e}")
            return [text]

        paraphrase_list = list(paraphrases)
        self.logger.info(f"Generated {len(paraphrase_list)} unique paraphrased variants.")
        return paraphrase_list if paraphrase_list else [text]