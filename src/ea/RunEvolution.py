## @file RunEvolution.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Main script for evolving LLM input prompts using mutation operators.

import json
import os
from .EvolutionEngine import EvolutionEngine
import nltk
from utils.custom_logging import get_logger

from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
population_path = project_root / "outputs" / "Population.json"

## @brief Main entry point: runs one evolution generation, applying selection and variation to prompts.
# @return None
def run_evolution(north_star_metric, log_file=None):
    # population_path = "../../outputs/Population.json"
    logger = get_logger("RunEvolution", log_file)
    logger.info(f"Starting evolution run using population file: {population_path}")

    if not population_path.exists():
        logger.error(f"Population file not found: {population_path}")
        raise FileNotFoundError(f"{population_path} not found.")

    with open(str(population_path), 'r', encoding='utf-8') as f:
        population = json.load(f)

    # Define status path for evolution status output
    status_path = project_root / "outputs" / "EvolutionStatus.json"

    logger.debug(f"Loaded {len(population)} genomes from {population_path}")
    logger.debug("Sorting population by prompt_id ASC, north_star_metric DESC, id ASC...")

    population.sort(key=lambda g: (
        g["prompt_id"],
        -(g.get(north_star_metric) if isinstance(g.get(north_star_metric), (int, float)) else 0.0),
        g["id"]
    ))
    
    with open(population_path, 'w', encoding='utf-8') as f:
        json.dump(population, f, indent=4)
    logger.info("Population re-saved in sorted order.")

    engine = EvolutionEngine(north_star_metric, log_file)
    engine.genomes = population
    engine.update_next_id()
    logger.debug(f"EvolutionEngine next_id set to {engine.next_id}")

    # Collect prompt evolution status (complete or pending_evolution)
    prompt_status = {}
    for genome in population:
        pid = genome["prompt_id"]
        if pid not in prompt_status:
            prompt_status[pid] = genome.get("status")
    logger.debug(f"Collected prompt status map for {len(prompt_status)} prompt_ids")

    completed_prompt_ids = {pid for pid, status in prompt_status.items() if status == "complete"}
    pending_prompt_ids = sorted(pid for pid, status in prompt_status.items() if status == "pending_evolution")

    logger.info(f"Completed prompt_ids (skipped): {sorted(completed_prompt_ids)}")
    logger.info(f"Pending prompt_ids (to process): {pending_prompt_ids}")

    # Track evolution status for all genomes
    evolution_status = []
    for genome in population:
        evolution_status.append({
            "generation": genome["generation"],
            "prompt_id": genome["prompt_id"],
            "status": genome.get("status", "unknown")
        })

    logger.debug(f"Initialized EvolutionEngine: starting next_id={engine.next_id}")

    for prompt_id in pending_prompt_ids:
        logger.info(f"Processing prompt_id={prompt_id}")
        logger.debug(f"Calling generate_variants() for prompt_id={prompt_id}")
        engine.generate_variants(prompt_id)
        logger.info(f"Completed variant generation for prompt_id={prompt_id}")

        with open(population_path, 'w', encoding='utf-8') as f:
            json.dump(engine.genomes, f, indent=4)
        logger.info(f"Saved updated population after processing prompt_id={prompt_id}")

    # Sort population before deduplication
    population = engine.genomes
    population.sort(key=lambda g: (
        g["prompt_id"],
        -(g.get(north_star_metric) if isinstance(g.get(north_star_metric), (int, float)) else 0.0),
        g["id"]
    ))
    logger.debug("Sorted population by prompt_id ASC, north_star_metric DESC, id ASC before deduplication.")

    from collections import defaultdict

    # Keep generation 0 genomes as-is
    gen_zero = [g for g in population if g["generation"] == 0]
    gen_gt_zero = [g for g in population if g["generation"] > 0]

    # Deduplicate gen > 0 by exact prompt string (case-insensitive), preserving sort order
    seen_prompts = set()
    deduplicated = []
    for genome in gen_gt_zero:
        norm_prompt = genome["prompt"].strip().lower()
        if norm_prompt not in seen_prompts:
            deduplicated.append(genome)
            seen_prompts.add(norm_prompt)

    # Final population = gen 0 + unique variants
    final_population = gen_zero + deduplicated

    # Sort final population again for consistency
    final_population.sort(key=lambda g: (
        g["prompt_id"],
        -(g.get(north_star_metric) if isinstance(g.get(north_star_metric), (int, float)) else 0.0),
        g["id"]
    ))

    logger.info(f"Deduplicated population: {len(population)} → {len(final_population)}")

    with open(population_path, 'w', encoding='utf-8') as f:
        json.dump(final_population, f, indent=4)
    logger.info("Population re-saved in sorted and deduplicated order.")

    # Save evolution status to JSON file
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(evolution_status, f, indent=4)
    logger.info(f"Saved evolution status for {len(evolution_status)} records to {status_path}")

