## @file TextVariationOperators.py
# @author Onkar Shelar (os9660@rit.edu)
# @brief Concrete mutation operators for prompt-level variations in evolutionary search.

import random
import torch
import spacy
from nltk.corpus import wordnet as wn
from typing import List
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    BertForMaskedLM,
)
from huggingface_hub import snapshot_download
from .VariationOperators import VariationOperator
from dotenv import load_dotenv
from itertools import combinations, product
import logging
logger = logging.getLogger(__name__)


from generator.LLaMaTextGenerator import LlaMaTextGenerator
generator = LlaMaTextGenerator(log_file=None)  #

load_dotenv()

nlp = spacy.load("en_core_web_sm")

SYNONYMS = {
    "good": ["great", "excellent", "nice"],
    "bad": ["terrible", "awful", "poor"],
    "people": ["individuals", "humans", "persons"],
    "problem": ["issue", "challenge", "difficulty"],
    "happy": ["joyful", "content", "cheerful"],
    "sad": ["unhappy", "miserable", "gloomy"],
}

# ## @class SynonymReplacementOperator
# # @brief Replaces selected words with synonyms from a predefined dictionary.
# class SynonymReplacementOperator(VariationOperator):
#     def __init__(self):
#         super().__init__("SynonymReplacement", "mutation", "Replaces a word with a synonym from a simple dictionary.")
#         logger.debug(f"Initialized operator: {self.name}")

#     def apply(self, text: str) -> str:
#         words = text.split()
#         candidates = [i for i, w in enumerate(words) if w.lower() in SYNONYMS]
#         if not candidates:
#             return text
#         idx = random.choice(candidates)
#         word = words[idx].lower()
#         words[idx] = random.choice(SYNONYMS[word])
#         logger.debug(f"{self.name}: Replaced word at index {idx} with '{words[idx]}'")
#         return " ".join(words)

## @class RandomDeletionOperator
# @brief Deletes a randomly selected word from the input text.
class RandomDeletionOperator(VariationOperator):
    def __init__(self):
        super().__init__("RandomDeletion", "mutation", "Deletes a random word.")
        logger.debug(f"Initialized operator: {self.name}")

    # def apply(self, text: str) -> str:
    #     words = text.split()
    #     if len(words) <= 1:
    #         return text
    #     idx = random.randint(0, len(words) - 1)
    #     del words[idx]
    #     logger.debug(f"{self.name}: Deleted word at index {idx}")
    #     return " ".join(words)
    
    def apply(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= 1:
            return [text]
        variants = []
        for idx in range(len(words)):
            variant = words[:idx] + words[idx+1:]
            variants.append(" ".join(variant))
        return variants

## @class WordShuffleOperator
# @brief Swaps two adjacent words in the input text.
class WordShuffleOperator(VariationOperator):
    def __init__(self):
        super().__init__("WordShuffle", "mutation", "Swaps two adjacent words.")
        logger.debug(f"Initialized operator: {self.name}")

    # def apply(self, text: str) -> str:
    #     words = text.split()
    #     if len(words) < 2:
    #         return text
    #     idx = random.randint(0, len(words) - 2)
    #     words[idx], words[idx + 1] = words[idx + 1], words[idx]
    #     logger.debug(f"{self.name}: Swapped words at indices {idx} and {idx + 1}")
    #     return " ".join(words)
    
    def apply(self, text: str) -> List[str]:
        words = text.split()
        if len(words) < 2:
            return [text]
        variants = []
        for i in range(len(words) - 1):
            swapped = words[:]
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            variants.append(" ".join(swapped))
        logger.debug(f"{self.name}: Generated {len(variants)} variants by adjacent swaps.")
        return variants
        
        

# ## @class CharacterSwapOperator
# # @brief Swaps two characters in a randomly selected word.
# class CharacterSwapOperator(VariationOperator):
#     def __init__(self):
#         super().__init__("CharacterSwap", "mutation", "Swaps two characters in a random word.")
#         logger.debug(f"Initialized operator: {self.name}")

#     def apply(self, text: str) -> str:
#         words = text.split()
#         idx = random.randint(0, len(words) - 1)
#         word = words[idx]
#         if len(word) < 2:
#             return text
#         chars = list(word)
#         j = random.randint(0, len(chars) - 2)
#         chars[j], chars[j + 1] = chars[j + 1], chars[j]
#         words[idx] = "".join(chars)
#         logger.debug(f"{self.name}: Swapped characters at positions {j} and {j + 1} in word index {idx}")
#         return " ".join(words)

## @class POSAwareSynonymReplacement
# @brief Uses spaCy POS tags and WordNet to replace a word with a context-aware synonym.
class POSAwareSynonymReplacement(VariationOperator):
    def __init__(self):
        super().__init__("POSAwareSynonymReplacement", "mutation", "WordNet synonym replacement based on spaCy POS.")
        logger.debug(f"Initialized operator: {self.name}")

    # def apply(self, text: str) -> str:
    #     for _ in range(3):
    #         doc = nlp(text)
    #         words = text.split()
    #         if len(words) != len(doc):
    #             words = [t.text for t in doc]

    #         replacements = []
    #         for token in doc:
    #             if token.pos_ in {"ADJ", "VERB", "NOUN", "ADV"}:
    #                 wn_pos = {"ADJ": wn.ADJ, "VERB": wn.VERB, "NOUN": wn.NOUN, "ADV": wn.ADV}[token.pos_]
    #                 synonyms = {
    #                     lemma.name().replace("_", " ")
    #                     for syn in wn.synsets(token.text, pos=wn_pos)
    #                     for lemma in syn.lemmas()
    #                     if lemma.name().lower() != token.text.lower()
    #                 }
    #                 if synonyms:
    #                     replacements.append((token.i, random.choice(list(synonyms))))

    #         if replacements:
    #             i, replacement = random.choice(replacements)
    #             if i < len(words):
    #                 mutated = words.copy()
    #                 mutated[i] = replacement
    #                 logger.debug(f"{self.name}: Replaced token at index {i} with '{replacement}'")
    #                 result = " ".join(mutated)
    #                 if result.lower().strip() != text.lower().strip():
    #                     return result

    #     return text

    def apply(self, text: str) -> List[str]:
        doc = nlp(text)
        words = text.split()
        if len(words) != len(doc):
            words = [t.text for t in doc]
        variants = set()

        # Group token indices by POS type
        pos_indices = {"ADJ": [], "VERB": [], "NOUN": [], "ADV": []}
        for token in doc:
            if token.pos_ in pos_indices:
                pos_indices[token.pos_].append(token.i)

        # Map to hold synonyms per token index
        synonym_map = {}

        for pos, indices in pos_indices.items():
            wn_pos = {"ADJ": wn.ADJ, "VERB": wn.VERB, "NOUN": wn.NOUN, "ADV": wn.ADV}[pos]
            for i in indices:
                token = doc[i]
                synonyms = {
                    lemma.name().replace("_", " ")
                    for syn in wn.synsets(token.text, pos=wn_pos)
                    for lemma in syn.lemmas()
                    if lemma.name().lower() != token.text.lower()
                }
                if synonyms:
                    synonym_map[i] = list(synonyms)

        # Individual replacements
        for i, syns in synonym_map.items():
            for synonym in syns:
                mutated = words.copy()
                mutated[i] = synonym
                variant = " ".join(mutated)
                if variant.lower().strip() != text.lower().strip():
                    variants.add(variant)

        # Multi-word combinations
        indices = list(synonym_map.keys())
        for r in range(2, len(indices) + 1):
            for combo in combinations(indices, r):
                synonym_lists = [synonym_map[i] for i in combo]
                for replacements in product(*synonym_lists):
                    mutated = words.copy()
                    for idx, repl in zip(combo, replacements):
                        mutated[idx] = repl
                    variant = " ".join(mutated)
                    if variant.lower().strip() != text.lower().strip():
                        variants.add(variant)

        return list(variants) if variants else [text]

## @class BertMLMOperator
# @brief Uses BERT Masked Language Model to replace one word with a predicted alternative.
class BertMLMOperator(VariationOperator):
    def __init__(self):
        super().__init__("BertMLM", "mutation", "Uses BERT MLM to replace one word.")
        logger.debug(f"Initialized operator: {self.name}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return [text]

        variants = set()
        for idx in range(len(words)):
            original = words[idx]
            masked_words = words.copy()
            masked_words[idx] = "[MASK]"
            masked_text = " ".join(masked_words)

            inputs = self.tokenizer(masked_text, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            topk = torch.topk(logits[0, mask_idx], k=5, dim=-1).indices[0].tolist()

            for token_id in topk:
                new_word = self.tokenizer.decode([token_id]).strip()
                mutated = words.copy()
                mutated[idx] = new_word
                result = " ".join(mutated).strip()
                if result.lower() != text.strip().lower():
                    variants.add(result)

        return list(variants) if variants else [text]

## @class LLMBasedParaphrasingOperator
# @brief Uses your LLM to paraphrase the input multiple times with north star guidance.
class LLMBasedParaphrasingOperator(VariationOperator):
    def __init__(self, generator, north_star_metric):
        super().__init__("LLMBasedParaphrasing", "mutation", "Uses LLM to paraphrase input multiple times with optimization intent.")
        self.generator = generator
        self.north_star_metric = north_star_metric
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> List[str]:
        variants = set()
        instruction = f"Paraphrase the following in a way that increases the {self.north_star_metric} (our north star metric in moderation) as much as possible:"

        for _ in range(4):
            full_prompt = f"{instruction}\n{text}"
            try:
                result = self.generator.generate_from_prompt(full_prompt)
                paraphrase = result.get("generated_response", "").strip()
                if paraphrase and paraphrase.lower() != text.lower():
                    variants.add(paraphrase)
                    logger.debug(f"{self.name}: Generated variant '{paraphrase}'")
            except Exception as e:
                logger.error(f"{self.name}: Failed to generate variant: {e}")

        return list(variants) if variants else [text]

## @class BackTranslationOperator
# @brief Performs back-translation via English-Hindi-English for paraphrasing.
class BackTranslationOperator(VariationOperator):
    def __init__(self):
        super().__init__("BackTranslation", "mutation", "Performs EN→HI→EN back-translation.")
        logger.debug(f"Initialized operator: {self.name}")
        # Ensure translation models are pre‑downloaded into the HF cache (strategy #1)
        for model_id in ("Helsinki-NLP/opus-mt-en-hi", "Helsinki-NLP/opus-mt-hi-en"):
            try:
                snapshot_download(model_id, local_files_only=True)
            except Exception:
                logger.info(f"Model {model_id} not found in cache. Downloading...")
                snapshot_download(model_id, local_files_only=False, resume_download=True)
        # Load translation pipelines from local cache
        en_hi_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-en-hi", local_files_only=True
        )
        en_hi_tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-hi", local_files_only=True
        )
        self.en_hi = pipeline(
            "translation_en_to_hi", model=en_hi_model, tokenizer=en_hi_tokenizer
        )

        hi_en_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-hi-en", local_files_only=True
        )
        hi_en_tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-hi-en", local_files_only=True
        )
        self.hi_en = pipeline(
            "translation_hi_to_en", model=hi_en_model, tokenizer=hi_en_tokenizer
        )

    def apply(self, text: str) -> List[str]:
        variants = set()
        attempts = 0
        while len(variants) < 4 and attempts < 10:
            try:
                hindi = self.en_hi(text, max_length=1024)[0]['translation_text']
                english = self.hi_en(hindi, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                cleaned = english.strip().lower()
                if cleaned and cleaned != text.strip().lower() and cleaned not in variants:
                    logger.debug(f"{self.name}: Back-translated to '{english}'")
                    variants.add(english.strip())
            except Exception as e:
                logger.error(f"[BackTranslation error]: {e}")
            attempts += 1
        return list(variants) if variants else [text]




# NOTE: Specify the north_star_metric as needed, e.g., "toxicity_score"
SINGLE_PARENT_OPERATORS = [
    POSAwareSynonymReplacement(),
    BertMLMOperator(),
    BackTranslationOperator()
]


# Returns the list of single-parent operators, including LLMBasedParaphrasingOperator with the provided north_star_metric.
def get_single_parent_operators(north_star_metric):
    return [
        POSAwareSynonymReplacement(),
        BertMLMOperator(),
        LLMBasedParaphrasingOperator(generator, north_star_metric),
        BackTranslationOperator()
    ]


## @class SentenceLevelCrossover
# @brief Combines sentences from two parent texts to create a new variant.
class SentenceLevelCrossover(VariationOperator):
    def __init__(self):
        super().__init__("SentenceLevelCrossover", "crossover", "Combines sentences from two parent texts.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> str:
        if not isinstance(parent_texts, list) or len(parent_texts) < 2:
            logger.warning(f"{self.name}: Insufficient parents for crossover.")
            return parent_texts[0] if parent_texts else ""

        parent1_sentences = parent_texts[0].split(". ")
        parent2_sentences = parent_texts[1].split(". ")

        # Combine half from each (fallback if fewer sentences exist)
        num_sentences_p1 = max(1, len(parent1_sentences) // 2)
        num_sentences_p2 = max(1, len(parent2_sentences) // 2)

        crossover_result = parent1_sentences[:num_sentences_p1] + parent2_sentences[:num_sentences_p2]
        result_text = ". ".join(crossover_result).strip()

        if not result_text.endswith("."):
            result_text += "."

        logger.debug(f"{self.name}: Created crossover result with {len(crossover_result)} sentences.")
        return result_text

MULTI_PARENT_OPERATORS = [
    SentenceLevelCrossover()
]

## @brief Returns the list of variation operators based on parent count.
# @param num_parents Number of parent genomes.
# @param north_star_metric (optional) Metric name for LLMBasedParaphrasingOperator.
# @return List of operator instances.
def get_applicable_operators(num_parents: int, north_star_metric):
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric)
    return MULTI_PARENT_OPERATORS