## @file EvolutionEngine.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Core logic for prompt mutation and evolutionary search

import random
from typing import List, Dict
from .TextVariationOperators import get_applicable_operators
import logging
logger = logging.getLogger(__name__)


## @class EvolutionEngine
class EvolutionEngine:

    ## @brief Initializes the EvolutionEngine with an empty genome list and ID counter.
    # @return None
    def __init__(self, north_star_metric):
        self.genomes: List[Dict] = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={self.north_star_metric}")

    ## @brief Updates the next available genome ID based on current population.
    # @return None
    def update_next_id(self):
        if self.genomes:
            self.next_id = max(g["id"] for g in self.genomes) + 1
        else:
            self.next_id = 0
        logger.debug(f"Updated next_id to {self.next_id}")

    def generate_variants(self, genomes: List[Dict]) -> List[Dict]:
        logger.debug(f"Generating variants from population of size {len(genomes)}")
        prompt_id_to_genomes = {}
        for g in genomes:
            prompt_id_to_genomes.setdefault(g["prompt_id"], []).append(g)

        offspring = []
        for prompt_id, prompt_genomes in prompt_id_to_genomes.items():
            prompt_genomes_sorted = sorted(
                prompt_genomes,
                key=lambda g: g.get(self.north_star_metric) or 0.0,
                reverse=True
            )
            if len(prompt_genomes_sorted) == 1:
                crossover_parents = None
                mutation_parent = prompt_genomes_sorted[0]
            elif 2 <= len(prompt_genomes_sorted) <= 4:
                crossover_parents = prompt_genomes_sorted
                mutation_parent = prompt_genomes_sorted[0]
            else:
                crossover_parents = prompt_genomes_sorted[:4]
                mutation_parent = prompt_genomes_sorted[0]

            existing_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == prompt_id)

            # Mutation
            mutation_operators = get_applicable_operators(1, self.north_star_metric)
            for op in mutation_operators:
                if op.operator_type != "mutation":
                    continue
                try:
                    variants = op.apply(mutation_parent["prompt"])
                    for vp in variants:
                        norm_vp = vp.strip().lower()
                        if norm_vp in existing_prompts:
                            continue
                        existing_prompts.add(norm_vp)
                        child = {
                            "id": self.next_id,
                            "prompt_id": prompt_id,
                            "prompt": vp,
                            "model_provider": None,
                            "model_name": None,
                            "generated_response": None,
                            "moderation_result": None,
                            "operator": op.name,
                            "parents": [mutation_parent["id"]],
                            "generation": mutation_parent["generation"] + 1,
                            "status": "pending_generation",
                            "creation_info": {
                                "type": "mutation",
                                "operator": op.name,
                                "source_generation": mutation_parent["generation"]
                            }
                        }
                        self.next_id += 1
                        logger.debug(f"Created mutation variant id={child['id']} for prompt_id={prompt_id}")
                        offspring.append(child)
                except Exception as e:
                    logger.error(f"[Mutation Error] {op.name}: {e}")

            # Crossover
            if crossover_parents:
                crossover_operators = get_applicable_operators(len(crossover_parents), self.north_star_metric)
                for op in crossover_operators:
                    if op.operator_type != "crossover":
                        continue
                    for parent in crossover_parents:
                        try:
                            variants = op.apply(parent["prompt"])
                            for vp in variants:
                                norm_vp = vp.strip().lower()
                                if norm_vp in existing_prompts:
                                    continue
                                existing_prompts.add(norm_vp)
                                child = {
                                    "id": self.next_id,
                                    "prompt_id": prompt_id,
                                    "prompt": vp,
                                    "model_provider": None,
                                    "model_name": None,
                                    "generated_response": None,
                                    "moderation_result": None,
                                    "operator": op.name,
                                    "parents": [parent["id"]],
                                    "generation": parent["generation"] + 1,
                                    "status": "pending_generation",
                                    "creation_info": {
                                        "type": "crossover",
                                        "operator": op.name,
                                        "source_generation": parent["generation"]
                                    }
                                }
                                self.next_id += 1
                                logger.debug(f"Created crossover variant id={child['id']} for prompt_id={prompt_id}")
                                offspring.append(child)
                        except Exception as e:
                            logger.error(f"[Crossover Error] {op.name}: {e}")

        return offspring