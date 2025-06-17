import sys
import time
import json
from utils.logging import get_logger, get_log_filename

logger = None

def main(model_names=None, max_generations=None):
    log_file = get_log_filename()
    print(f"DEBUG: log_file path resolved to: {log_file}")
    logger = get_logger("main", log_file)
    logger.debug(f"Logger initialized in main with log_file: {log_file}")

    start_time = time.time()
    north_star_metric = "violence"  # Primary optimization metric for evolution
    logger.debug(f"North star metric for optimization set to: {north_star_metric}")
    logger.debug(f"Maximum generations allowed for evolution: {max_generations}")

    logger.info("Initializing pipeline...")

    import os
    from utils.initialize_population import load_and_initialize_population

    if not os.path.exists("outputs/Population.json"):
        try:
            population_start = time.time()
            logger.info("Population file not found. Initializing population from prompt.xlsx...")
            load_and_initialize_population(
                input_path="data/prompt.xlsx",
                output_path="outputs/Population.json",
                log_file=log_file
            )
            logger.info("Population successfully initialized and saved.")
            logger.info(f"Population initialization completed in {time.time() - population_start:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to initialize population: {e}")
            return
    else:
        logger.info("Existing population found. Skipping initialization.")

    while True:
        try:
            generation_start = time.time()
            from generator.LLaMaTextGenerator import LlaMaTextGenerator
            logger.info("Generating responses using LLaMA model...")
            generator = LlaMaTextGenerator(log_file=log_file)
            generator.process_population()
            logger.info("Text generation completed and population updated.")
            logger.info(f"Text generation completed in {time.time() - generation_start:.2f} seconds.")
        except Exception as e:
            logger.error(f"Generation failed: {e}")

        try:
            evaluation_start = time.time()
            from evaluator.openai_moderation import run_moderation_on_population
            logger.info("Evaluating generated responses using OpenAI moderation API...")
            run_moderation_on_population(
                pop_path="outputs/Population.json",
                log_file=log_file
            )
            logger.info("Evaluation completed and population updated with moderation scores.")
            logger.info(f"Evaluation completed in {time.time() - evaluation_start:.2f} seconds.")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

        try:
            evolution_start = time.time()
            from ea.RunEvolution import run_evolution
            logger.info("Running evolution on population...")
            run_evolution(
                north_star_metric=north_star_metric,
                log_file=log_file
            )
            logger.info("Evolution process completed and population updated.")
            logger.info(f"Evolution completed in {time.time() - evolution_start:.2f} seconds.")
        except Exception as e:
            logger.error(f"Evolution failed: {e}")

        try:
            post_evolution_generation_start = time.time()
            logger.info("Generating responses for new variants post-evolution...")
            try:
                with open("outputs/Population.json", "r", encoding="utf-8") as f:
                    population = json.load(f)

                # Filter out genomes for prompt_ids with complete status
                prompt_status_map = {g["prompt_id"]: g["status"] for g in population if g["generation"] == 0}
                incomplete_ids = {pid for pid, status in prompt_status_map.items() if status != "complete"}

                generator = LlaMaTextGenerator(log_file=log_file)

                for i, genome in enumerate(population):
                    if genome["prompt_id"] in incomplete_ids and genome["status"] == "pending_generation":
                        logger.info(f"Generating for genome_id={genome['id']}...")
                        try:
                            generator.generate_for_genome(genome)
                            # genome["status"] = "pending_evaluation"
                            logger.info(f"Evaluating for genome_id={genome['id']}...")
                            run_moderation_on_population(
                                pop_path="outputs/Population.json",
                                single_genome=genome,
                                log_file=log_file
                            )
                            # Update the genome in the population
                            population[i] = genome
                            with open("outputs/Population.json", "w", encoding="utf-8") as f:
                                json.dump(population, f, indent=4)
                        except Exception as e:
                            logger.error(f"Generation or Evaluation failed for genome_id={genome['id']}: {e}")

                logger.info("Post-evolution generation and evaluation completed.")
            except Exception as e:
                logger.error(f"Post-evolution generation/evaluation failed: {e}")
            finally:
                logger.info(f"Post-evolution generation completed in {time.time() - post_evolution_generation_start:.2f} seconds.")
        except Exception as outer_e:
            logger.error(f"Unexpected failure in post-evolution generation block: {outer_e}")

        try:
            with open("outputs/Population.json", "r", encoding="utf-8") as f:
                population = json.load(f)
            generations = [g["generation"] for g in population]
            statuses = {g["prompt_id"]: g["status"] for g in population}
            max_gen = max(generations)
            pending_exists = any(status != "complete" for status in statuses.values())

            if max_generations is not None:
                if max_gen >= max_generations:
                    logger.info(f"Maximum generation limit ({max_generations}) reached. Stopping pipeline.")
                    break
            if not pending_exists:
                logger.info("All prompt_ids have reached complete status. Stopping pipeline.")
                break

            logger.info("Continuing pipeline loop: prompts pending and generation limit not reached.")
        except Exception as e:
            logger.error(f"Failed to determine stopping condition: {e}")
            break

    logger.info(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=None, help="Maximum number of evolution generations. If not set, runs until all prompts are complete.")
    parser.add_argument("model_names", nargs="*", default=[])
    args = parser.parse_args()
    main(model_names=args.model_names, max_generations=args.generations)