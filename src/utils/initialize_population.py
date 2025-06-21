from utils.custom_logging import get_logger
from pathlib import Path
import pandas as pd
import os
import json
import time
from typing import List, Dict, Any, Optional
from utils.custom_logging import PerformanceLogger
import logging

def load_and_initialize_population(input_path: str, output_path: str, log_file: Optional[str] = None) -> None:
    """Load prompts from Excel file and initialize population with comprehensive logging"""
    logger = get_logger("initialize_population", log_file)
    
    with PerformanceLogger(logger, "Initialize Population", input_path=input_path, output_path=output_path):
        try:
            logger.info("Starting population initialization")
            logger.info("Input file: %s", input_path)
            logger.info("Output file: %s", output_path)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                logger.error("Input file not found: %s", input_path)
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Load Excel file
            with PerformanceLogger(logger, "Load Excel File"):
                try:
                    logger.debug("Loading Excel file: %s", input_path)
                    df = pd.read_excel(input_path)
                    logger.info("Successfully loaded Excel file with %d rows and %d columns", 
                               len(df), len(df.columns))
                    logger.debug("Column names: %s", list(df.columns))
                except Exception as e:
                    logger.error("Failed to load Excel file: %s", e, exc_info=True)
                    raise
            
            # Extract prompts
            with PerformanceLogger(logger, "Extract Prompts"):
                try:
                    # Look for common column names that might contain prompts
                    prompt_columns = ['prompt', 'text', 'input', 'query', 'instruction', 'content']
                    prompt_column = None
                    
                    for col in prompt_columns:
                        if col in df.columns:
                            prompt_column = col
                            break
                    
                    if prompt_column is None:
                        # If no standard column found, use the first text-like column
                        for col in df.columns:
                            if df[col].dtype == 'object' and len(df[col].iloc[0]) > 10:
                                prompt_column = col
                                break
                    
                    if prompt_column is None:
                        logger.error("No suitable prompt column found in Excel file")
                        logger.debug("Available columns: %s", list(df.columns))
                        raise ValueError("No suitable prompt column found")
                    
                    logger.info("Using column '%s' for prompts", prompt_column)
                    
                    # Extract prompts
                    prompts = df[prompt_column].dropna().tolist()
                    logger.info("Extracted %d prompts from column '%s'", len(prompts), prompt_column)
                    
                    # Log some sample prompts
                    for i, prompt in enumerate(prompts[:3]):
                        logger.debug("Sample prompt %d: %s", i + 1, prompt[:100] + "..." if len(prompt) > 100 else prompt)
                    
                except Exception as e:
                    logger.error("Failed to extract prompts: %s", e, exc_info=True)
                    raise
            
            # Create population
            with PerformanceLogger(logger, "Create Population"):
                try:
                    population = []
                    start_time = time.time()
                    
                    for i, prompt in enumerate(prompts):
                        genome = {
                            'id': str(i + 1),
                            'prompt_id': i,
                            'prompt': str(prompt).strip(),
                            'generation': 0,
                            'status': 'pending_generation',
                            'created_timestamp': time.time(),
                            'source_file': input_path,
                            'source_column': prompt_column
                        }
                        population.append(genome)
                    
                    creation_time = time.time() - start_time
                    logger.info("Created %d genomes in %.3f seconds", len(population), creation_time)
                    
                    # Log population statistics
                    prompt_lengths = [len(g['prompt']) for g in population]
                    avg_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
                    min_length = min(prompt_lengths) if prompt_lengths else 0
                    max_length = max(prompt_lengths) if prompt_lengths else 0
                    
                    logger.info("Population statistics:")
                    logger.info("  - Average prompt length: %.1f characters", avg_length)
                    logger.info("  - Min prompt length: %d characters", min_length)
                    logger.info("  - Max prompt length: %d characters", max_length)
                    
                except Exception as e:
                    logger.error("Failed to create population: %s", e, exc_info=True)
                    raise
            
            # Save population
            with PerformanceLogger(logger, "Save Population"):
                try:
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(population, f, indent=2, ensure_ascii=False)
                    
                    logger.info("Successfully saved population to %s", output_path)
                    
                    # Log file size
                    file_size = os.path.getsize(output_path)
                    logger.debug("Output file size: %d bytes", file_size)
                    
                except Exception as e:
                    logger.error("Failed to save population: %s", e, exc_info=True)
                    raise
            
            # Validate saved population
            with PerformanceLogger(logger, "Validate Population"):
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        saved_population = json.load(f)
                    
                    if len(saved_population) != len(population):
                        logger.error("Population size mismatch: expected %d, got %d", 
                                   len(population), len(saved_population))
                        raise ValueError("Population size mismatch")
                    
                    logger.info("Population validation successful: %d genomes saved", len(saved_population))
                    
                except Exception as e:
                    logger.error("Failed to validate saved population: %s", e, exc_info=True)
                    raise
            
            # Log final summary
            total_time = time.time() - time.time()  # This will be 0, but tracks the operation
            logger.info("Population initialization completed successfully:")
            logger.info("  - Input file: %s", input_path)
            logger.info("  - Output file: %s", output_path)
            logger.info("  - Prompts processed: %d", len(prompts))
            logger.info("  - Genomes created: %d", len(population))
            logger.info("  - Generation: 0 (initial)")
            
        except Exception as e:
            logger.error("Population initialization failed: %s", e, exc_info=True)
            raise

def validate_population_file(population_path: str, log_file: Optional[str] = None) -> Dict[str, Any]:
    """Validate a population file and return statistics with comprehensive logging"""
    logger = get_logger("validate_population", log_file)
    
    with PerformanceLogger(logger, "Validate Population File", file_path=population_path):
        try:
            logger.info("Validating population file: %s", population_path)
            
            if not os.path.exists(population_path):
                logger.error("Population file not found: %s", population_path)
                raise FileNotFoundError(f"Population file not found: {population_path}")
            
            # Load population
            with PerformanceLogger(logger, "Load Population for Validation"):
                try:
                    with open(population_path, 'r', encoding='utf-8') as f:
                        population = json.load(f)
                    logger.info("Successfully loaded population with %d genomes", len(population))
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse population JSON: %s", e, exc_info=True)
                    raise
                except Exception as e:
                    logger.error("Failed to load population: %s", e, exc_info=True)
                    raise
            
            # Analyze population
            with PerformanceLogger(logger, "Analyze Population"):
                try:
                    stats = {
                        'total_genomes': len(population),
                        'generations': set(),
                        'statuses': {},
                        'prompt_ids': set(),
                        'prompt_lengths': [],
                        'errors': []
                    }
                    
                    for genome in population:
                        # Check required fields
                        required_fields = ['id', 'prompt_id', 'prompt', 'generation', 'status']
                        for field in required_fields:
                            if field not in genome:
                                stats['errors'].append(f"Missing required field '{field}' in genome {genome.get('id', 'unknown')}")
                        
                        # Collect statistics
                        stats['generations'].add(genome.get('generation', -1))
                        stats['prompt_ids'].add(genome.get('prompt_id', -1))
                        
                        status = genome.get('status', 'unknown')
                        stats['statuses'][status] = stats['statuses'].get(status, 0) + 1
                        
                        prompt_length = len(genome.get('prompt', ''))
                        stats['prompt_lengths'].append(prompt_length)
                    
                    # Calculate additional statistics
                    if stats['prompt_lengths']:
                        stats['avg_prompt_length'] = sum(stats['prompt_lengths']) / len(stats['prompt_lengths'])
                        stats['min_prompt_length'] = min(stats['prompt_lengths'])
                        stats['max_prompt_length'] = max(stats['prompt_lengths'])
                    
                    stats['generations'] = sorted(list(stats['generations']))
                    stats['prompt_ids'] = sorted(list(stats['prompt_ids']))
                    
                    logger.info("Population analysis completed:")
                    logger.info("  - Total genomes: %d", stats['total_genomes'])
                    logger.info("  - Generations: %s", stats['generations'])
                    logger.info("  - Unique prompt IDs: %d", len(stats['prompt_ids']))
                    logger.info("  - Status distribution: %s", stats['statuses'])
                    
                    if stats['prompt_lengths']:
                        logger.info("  - Average prompt length: %.1f characters", stats['avg_prompt_length'])
                        logger.info("  - Min/Max prompt length: %d/%d characters", 
                                   stats['min_prompt_length'], stats['max_prompt_length'])
                    
                    if stats['errors']:
                        logger.warning("Found %d validation errors:", len(stats['errors']))
                        for error in stats['errors']:
                            logger.warning("  - %s", error)
                    else:
                        logger.info("No validation errors found")
                    
                    return stats
                    
                except Exception as e:
                    logger.error("Failed to analyze population: %s", e, exc_info=True)
                raise

        except Exception as e:
            logger.error("Population validation failed: %s", e, exc_info=True)
            raise