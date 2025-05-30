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
    def __init__(self):
        self.genomes: List[Dict] = []
        self.next_id = 0
        logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}")

    ## @brief Updates the next available genome ID based on current population.
    # @return None
    def update_next_id(self):
        if self.genomes:
            self.next_id = max(g["id"] for g in self.genomes) + 1
        else:
            self.next_id = 0
        logger.debug(f"Updated next_id to {self.next_id}")

    ## @brief Selects parent genomes via tournament selection.
    # @param genomes List of candidate genomes.
    # @param k Number of parents to select.
    # @param tournament_size Number of genomes per tournament.
    # @return List of selected parent genomes.
    def tournament_selection(self, genomes: List[dict], k: int, tournament_size: int = 5) -> List[dict]:
        if len(genomes) == 1:
            return [genomes[0]] * k

        selected_parents = []
        for _ in range(k):
            candidates = random.sample(genomes, min(tournament_size, len(genomes)))
            winner = max(candidates, key=lambda g: g.get("toxicity_score") or 0.0)
            selected_parents.append(winner)

        logger.debug(f"Tournament selection: selected {len(selected_parents)} parents from {len(genomes)} candidates (tournament_size={tournament_size})")
        return selected_parents

    ## @brief Generates offspring by applying variation operators to parents.
    # @param parents List of parent genome dictionaries.
    # @param num_offspring_per_operator Offspring per operator.
    # @return List of new offspring genome dictionaries.
    def generate_offspring(self, parents: List[Dict], num_offspring_per_operator: int = 20) -> List[Dict]:
        logger.debug(f"Generating offspring for {len(parents)} parents with {num_offspring_per_operator} offspring per operator")
        operators = get_applicable_operators(len(parents))
        offspring = []

        for operator in operators:
            for _ in range(num_offspring_per_operator):
                parent = random.choice(parents)

                try:
                    variant_prompt = operator.apply(parent["prompt"])
                    if variant_prompt.strip().lower() in set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == parent["prompt_id"]):
                        continue
                    seen_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == parent["prompt_id"])
                    seen_prompts.add(variant_prompt.strip().lower())
                except Exception as e:
                    logger.error(f"[Variation Failed] {operator.name}: {e}")
                    continue

                child = {
                    "id": self.next_id,
                    "prompt_id": parent["prompt_id"],
                    "prompt": variant_prompt,
                    "generated_response": None,
                    "toxicity_score": None,
                    "severe_toxicity_score": None,
                    "insult_score": None,
                    "operator": operator.name,
                    "parents": [parent["id"]],
                    "generation": parent["generation"] + 1,
                    "status": "pending_generation",
                    "creation_info": {
                        "type": operator.operator_type,
                        "operator": operator.name,
                        "source_generation": parent["generation"]
                    }
                }

                self.next_id += 1
                logger.debug(f"Created child id={self.next_id} for prompt_id={parent['prompt_id']} using operator {operator.name}")
                offspring.append(child)

        return offspring