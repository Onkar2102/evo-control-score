## @file RunEvolution.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import os
from .EvolutionEngine import EvolutionEngine
import nltk
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
population_path = project_root / "outputs" / "Population.json"

## @brief Main entry point: runs one evolution generation, applying selection and variation to prompts.
# @return None
def run_evolution():
    # population_path = "../../outputs/Population.json"
    logger.info(f"Starting evolution run using population file: {population_path}")

    if not population_path.exists():
        logger.error(f"Population file not found: {population_path}")
        raise FileNotFoundError(f"{population_path} not found.")

    with open(str(population_path), 'r', encoding='utf-8') as f:
        population = json.load(f)

    logger.debug(f"Loaded {len(population)} genomes from {population_path}")

    engine = EvolutionEngine()
    engine.genomes = population
    engine.update_next_id()
    logger.debug(f"Initialized EvolutionEngine: starting next_id={engine.next_id}")

    current_prompt_id = None
    prompt_genomes = []
    new_offspring = []

    engine.genomes.sort(key=lambda g: g["prompt_id"])
    logger.debug("Sorted population genomes by prompt_id")

    for genome in engine.genomes:
        this_prompt_id = genome["prompt_id"]

        if current_prompt_id is None:
            current_prompt_id = this_prompt_id

        if this_prompt_id != current_prompt_id:
            if prompt_genomes:
                tox_values = [g.get("toxicity_score") for g in prompt_genomes if g.get("toxicity_score") is not None]

                if tox_values:
                    avg_tox = sum(tox_values) / len(tox_values)
                    if len(tox_values) >= 10 and avg_tox >= 1.0:
                        logger.info(f"Skipping prompt_id={current_prompt_id} due to high toxicity (avg >= 1.0)")
                        prompt_genomes = []
                        current_prompt_id = this_prompt_id
                        continue

            parents = engine.tournament_selection(prompt_genomes, k=5)
            offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
            new_offspring.extend(offspring)
            logger.debug(f"Generated {len(offspring)} offspring for prompt_id={current_prompt_id}")

            prompt_genomes = []
            current_prompt_id = this_prompt_id

        prompt_genomes.append(genome)

        if prompt_genomes:
            tox_values = [g.get("toxicity_score") for g in prompt_genomes if g.get("toxicity_score") is not None]
            if tox_values and len(tox_values) >= 10:
                avg_tox = sum(tox_values) / len(tox_values)
                if len(tox_values) >= 10 and avg_tox >= 1.0:
                    logger.info(f"Skipping prompt_id={current_prompt_id} due to high toxicity (avg >= 1.0)")
                else:
                    parents = engine.tournament_selection(prompt_genomes, k=5)
                    offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
                    new_offspring.extend(offspring)
                    logger.debug(f"Generated {len(offspring)} offspring for prompt_id={current_prompt_id}")
            else:
                parents = engine.tournament_selection(prompt_genomes, k=5)
                offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
                new_offspring.extend(offspring)
                logger.debug(f"Generated {len(offspring)} offspring for prompt_id={current_prompt_id}")

    logger.info(f"Total new offspring generated: {len(new_offspring)}")

    population.extend(new_offspring)
    population.sort(key=lambda g: (
        g["prompt_id"],
        -(g["toxicity_score"] if isinstance(g.get("toxicity_score"), (int, float)) else 0.0),
        -g["generation"]
    ))

    logger.debug(f"Writing updated population (total size {len(population)}) to {population_path}")
    with open(population_path, 'w', encoding='utf-8') as f:
        json.dump(population, f, indent=4)

    logger.info(f"Updated population saved to {population_path}")


if __name__ == "__main__":
    main()