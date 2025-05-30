from utils.logging import get_logger, get_log_filename
from utils.initialize_population import load_and_initialize_population
from generator.Factory import get_generator, MODEL_MAPPING
import json
import sys

logger = None  # Will be initialized inside main()

def main(model_names=None):
    log_file = get_log_filename()
    logger = get_logger("main", log_file)

    logger.info("Initializing pipeline...")

    try:
        logger.info("Loading prompts and initializing population...")
        load_and_initialize_population(
            input_path="data/prompt.xlsx",
            output_path="outputs/Population.json",
            log_file=log_file
        )
        logger.info("Population successfully initialized and saved.")
    except Exception as e:
        logger.error(f"Failed to initialize population: {e}")
        return

    if model_names is None or not model_names:
        model_names = list(MODEL_MAPPING.keys())

    with open("outputs/Population.json", "r") as f:
        population = json.load(f)

    if not population:
        logger.error("Population is empty. No genomes found.")
        raise ValueError("No genome data available for generation.")

    updated = False
    generators = []

    if model_names:
        for model_alias in model_names:
            try:
                generator, model_metadata = get_generator(model_alias, log_file=log_file)
                generators.append((generator, model_metadata, model_alias))
            except Exception as e:
                logger.error(f"Skipping model '{model_alias}' due to error: {e}")
    else:
        try:
            all_generators = get_generator(log_file=log_file)
            generators.extend([
                (gen, meta, alias)
                for gen, meta in all_generators
                for alias in [meta.get("model_name", "unknown")]
            ])
        except Exception as e:
            logger.error(f"Failed to load all generators: {e}")

    # Prepare variables for new genomes and next ID
    new_genomes = []
    next_id = max([g["id"] for g in population]) + 1 if population else 0

    for genome in population:
        if genome["status"] == "pending_generation":
            for generator, model_metadata, model_name in generators:
                try:
                    response = generator.generate(genome["prompt"])
                    genome_copy = genome.copy()
                    genome_copy["generated_response"] = response
                    genome_copy["status"] = "pending_evaluation"
                    genome_copy["model_used"] = model_name

                    # Update prompt_id and genome ID properly if you're saving as new entries
                    genome_copy["id"] = next_id
                    next_id += 1
                    new_genomes.append(genome_copy)

                    logger.info(f"Generated response for genome ID {genome['id']} using model '{model_name}'")

                except Exception as e:
                    logger.error(f"Error generating response for genome ID {genome['id']} using model '{model_name}': {e}")

    # Add new genomes to population after generation loop
    if new_genomes:
        population.extend(new_genomes)
        updated = True

    if updated:
        with open("outputs/Population.json", "w") as f:
            json.dump(population, f, indent=2)
        logger.info("Updated population saved with generated responses.")
    else:
        logger.info("No genome required generation. Skipping generation step.")

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    # Example: python main.py llama3.2 openai
    args = sys.argv[1:]
    main(model_names=args)