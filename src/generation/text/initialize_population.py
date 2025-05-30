## @file initialize_population.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Initializes a structured genome population JSON from scored prompt-response pairs.
#
# This script processes a scored responses JSON file to assign unique prompt IDs
# and formats the data into a genome structure compatible with evolutionary mutation
# pipelines. The output is sorted and saved for use in prompt evolution.

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

## @brief Loads scored prompt‑response pairs, assigns unique prompt IDs, and
#         builds an initial genome population.
# @param input_path  Path to the JSON file containing scored prompt‑response entries.
# @param output_path Path where the initialized genome JSON will be saved.
def load_and_initialize_population(input_path: str, output_path: str):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded {len(data)} entries from {input_path}")
        logger.debug(f"Creating genome entries from loaded data.")
    except Exception as e:
        logger.error(f"Error loading file {input_path}: {e}")
        raise RuntimeError(f"Error loading file: {e}") from e

    prompt_to_id = {}
    prompt_counter = 0
    genomes = []

    for idx, entry in enumerate(data):
        prompt = entry["prompt"]

        if prompt not in prompt_to_id:
            prompt_to_id[prompt] = prompt_counter
            prompt_counter += 1

        genome = {
            "id": idx,
            "prompt_id": prompt_to_id[prompt] + 1,
            "prompt": prompt,
            "generated_response": entry["generated_response"],
            "toxicity_score": entry.get("toxicity_score"),
            "severe_toxicity_score": entry.get("severe_toxicity_score"),
            "insult_score": entry.get("insult_score"),
            "operator": "original",
            "parents": [],
            "generation": 0,
            "status": "scored"
        }

        genomes.append(genome)
    logger.debug(f"Created {len(genomes)} genome entries before sorting.")

    sorted_genomes = sorted(genomes, key=lambda g: g["prompt_id"])
    logger.debug(f"Sorted genomes by prompt_id; total sorted: {len(sorted_genomes)}.")

    logger.debug(f"Ensuring output directory exists for path: {output_path}")
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with output_path_obj.open("w", encoding="utf-8") as f:
            json.dump(sorted_genomes, f, indent=4)
        logger.debug(f"Wrote sorted population to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write population to {output_path}: {e}")
        raise

    logger.info(
        f"Initialized {len(sorted_genomes)} genomes from {len(prompt_to_id)} unique prompts"
    )
    logger.info(f"Saved sorted population to {output_path}")

##
# @section genome_schema Genome JSON schema
# Each genome dict has the following keys:
#  - id: unique integer
#  - prompt_id: group identifier (1-based)
#  - prompt: the text prompt
#  - generated_response: the LLM response text
#  - toxicity_score: float
#  - severe_toxicity_score: float
#  - insult_score: float
#  - operator: name of operator that generated this genome
#  - parents: list of parent genome IDs
#  - generation: generation number (0 for original)
#  - status: processing status string


## @brief CLI entry point which computes default paths and calls population initialization.
# @return None
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir.joinpath("../../../outputs/responses/generated_responses_with_scores.json").resolve()
    output_file = base_dir.joinpath("../../../outputs/Population.json").resolve()
    load_and_initialize_population(str(input_file), str(output_file))