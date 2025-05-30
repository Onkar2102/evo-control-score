from generator.LLaMaTextGenerator import LlamaTextGenerator
from .OpenAITextGenerator import OpenAITextGenerator
from .MistralTextGenerator import MistralTextGenerator
from utils.config import load_config
from utils.logging import get_logger


MODEL_MAPPING = {
    "llama": LlamaTextGenerator,
    "openai": OpenAITextGenerator,
    "mistral": LlamaTextGenerator,  # Mistral uses same class as LLaMA for now
}

def get_generator(name: str | None, log_file: str | None):
    logger = get_logger("Factory", log_file)
    if name is None:
        logger.info("No model name specified. Fetching all generators.")
        return get_all_generators()

    alias = name.lower()
    logger.info(f"Fetching generator for model alias: {alias}")

    config = load_config()
    model_configs = config.get("models", {})

    if alias not in MODEL_MAPPING:
        logger.error(f"Unknown model alias: '{alias}'")
        raise ValueError(f"Unknown model alias: '{alias}'. Expected one of: {list(MODEL_MAPPING.keys())}")

    if alias not in model_configs:
        logger.error(f"Model configuration for key '{alias}' not found in model_config.yaml.")
        raise ValueError(f"Model configuration for key '{alias}' not found in model_config.yaml.")

    generator_class = MODEL_MAPPING[alias]
    model_cfg = model_configs[alias]

    logger.debug(f"Loaded configuration for model alias '{alias}': {model_cfg}")
    metadata = {
        "alias": alias,
        "model_name": model_cfg.get("name", ""),
        "strategy": model_cfg.get("strategy", ""),
        "task_type": model_cfg.get("task_type", ""),
    }
    return generator_class(config=model_cfg), metadata

def get_all_generators(log_file: str | None):
    logger = get_logger("Factory", log_file)
    logger.info("Fetching all model generators...")
    config = load_config()
    model_configs = config.get("models", {})
    generators = []

    for alias, generator_class in MODEL_MAPPING.items():
        model_cfg = model_configs.get(alias)
        if not model_cfg:
            logger.warning(f"No configuration found for alias '{alias}'. Skipping.")
            continue

        metadata = {
            "alias": alias,
            "model_name": model_cfg.get("name", ""),
            "strategy": model_cfg.get("strategy", ""),
            "task_type": model_cfg.get("task_type", ""),
        }
        logger.debug(f"Initializing generator for alias '{alias}' with config: {model_cfg}")
        generators.append((generator_class(config=model_cfg), metadata))

    logger.info(f"Initialized {len(generators)} model generators.")
    return generators