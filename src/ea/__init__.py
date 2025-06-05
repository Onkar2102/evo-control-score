## @file src/ea/__init__.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Evolutionary Algorithm (EA) package for LLM toxicity optimization.
#
# This package provides:
#  - EvolutionEngine: the core EA loop (selection + variation)
#  - run_evolution: driver for one EA generation
#  - TextVariationOperators: concrete mutation operators
#  - get_applicable_operators: helper to pick operators based on parent count

from .EvolutionEngine import EvolutionEngine
from .RunEvolution import run_evolution
from .TextVariationOperators import (
    RandomDeletionOperator,
    WordShuffleOperator,
    POSAwareSynonymReplacement,
    BertMLMOperator,
    BackTranslationOperator,
    LLMBasedParaphrasingOperator,
    SentenceLevelCrossover,
    get_applicable_operators,
)

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "EvolutionEngine",
    "run_evolution",
    "RandomDeletionOperator",
    "WordShuffleOperator",
    "POSAwareSynonymReplacement",
    "BertMLMOperator",
    "BackTranslationOperator",
    "LLMBasedParaphrasingOperator",
    "SentenceLevelCrossover",
    "get_applicable_operators",
]