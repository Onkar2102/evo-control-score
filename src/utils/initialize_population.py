import json
from utils.logging import get_logger
from pathlib import Path
import pandas as pd

def load_and_initialize_population(input_path: str, output_path: str, log_file: str = None):
    logger = get_logger("initialize_population", log_file)
    logger.info(f"Reading prompts from Excel file: {Path(input_path).resolve()}")
    try:
        df = pd.read_excel(input_path)
        prompts = df["prompt"].dropna().unique().tolist()
        logger.debug(f"Loaded {len(prompts)} unique prompts from {input_path}")
        logger.info(f"First 3 prompts loaded: {prompts[:3]}")
    except Exception as e:
        logger.error(f"Error reading Excel file {input_path}: {e}")
        raise RuntimeError(f"Error reading Excel file: {e}") from e

    genomes = []
    for idx, prompt in enumerate(prompts):
        genome = {
            "id": idx,
            "prompt_id": idx + 1,
            "prompt": prompt,
            "generation": 0,
            "status": "pending_generation"
        }
        genomes.append(genome)
    logger.debug(f"Created {len(genomes)} genome entries before sorting.")

    sorted_genomes = sorted(genomes, key=lambda g: g["prompt_id"])
    logger.debug(f"Sorted genomes by prompt_id; total sorted: {len(sorted_genomes)}.")

    output_path_obj = Path(output_path)
    logger.info(f"Saving initialized population to: {output_path_obj.resolve()}")
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with output_path_obj.open("w", encoding="utf-8") as f:
            json.dump(sorted_genomes, f, indent=4)
        logger.debug(f"Wrote sorted population to {output_path}")
        logger.info(f"Saved {len(sorted_genomes)} genome entries to {output_path_obj.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write population to {output_path}: {e}")
        raise

    logger.info(
        f"Initialized {len(sorted_genomes)} genomes from {len(prompts)} unique prompts"
    )
    logger.info(f"Saved sorted population to {output_path}")