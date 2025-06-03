from utils.logging import get_logger, get_log_filename
from utils.initialize_population import load_and_initialize_population
import sys
from generator.LLaMaTextGenerator import LlaMaTextGenerator
from evaluator.openai_moderation import run_moderation_on_population
import time

logger = None  # Will be initialized inside main()

def main(model_names=None):
    log_file = get_log_filename()
    logger = get_logger("main", log_file)

    start_time = time.time()
    north_star_metric = "hate"  # Primary optimization metric for evolution
    logger.debug(f"North star metric for optimization set to: {north_star_metric}")

    logger.info("Initializing pipeline...")

    try:
        population_start = time.time()
        logger.info("Loading prompts and initializing population...")
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
    
    try:
        generation_start = time.time()
        logger.info("Generating responses using LLaMA model...")
        generator = LlaMaTextGenerator(log_file=log_file)
        generator.process_population()
        logger.info("Text generation completed and population updated.")
        logger.info(f"Text generation completed in {time.time() - generation_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

    try:
        evaluation_start = time.time()
        logger.info("Evaluating generated responses using OpenAI moderation API...")
        run_moderation_on_population(
            pop_path="outputs/Population.json",
            log_file=log_file
        )
        logger.info("Evaluation completed and population updated with moderation scores.")
        logger.info(f"Evaluation completed in {time.time() - evaluation_start:.2f} seconds.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    # try:
    #     logger.info("Extracting tokens and embeddings for prompts and responses...")
    #     embedding_start = time.time()
    #     generator.extract_embeddings_for_population(
    #         pop_path="outputs/Population.json",
    #         log_file=log_file
    #     )
    #     logger.info("Embedding extraction completed and population updated.")
    #     logger.info(f"Embedding extraction completed in {time.time() - embedding_start:.2f} seconds.")
    # except Exception as e:
    #     logger.error(f"Embedding extraction failed: {e}")

    # try:
    #     token_conversion_start = time.time()
    #     logger.info("Converting prompt and generated response to token IDs...")
    #     generator.convert_population_texts_to_tokens()
    #     logger.info("Token conversion completed and population updated.")
    #     logger.info(f"Token conversion completed in {time.time() - token_conversion_start:.2f} seconds.")
    # except Exception as e:
    #     logger.error(f"Token conversion failed: {e}")

    logger.info(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Example: python main.py llama3.2 openai
    args = sys.argv[1:]
    main(model_names=args)