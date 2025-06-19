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
from utils.logging import get_logger
from openai import OpenAI
import os

# openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key securely


from generator.LLaMaTextGenerator import LlaMaTextGenerator
generator = LlaMaTextGenerator(log_file=None)

load_dotenv()

nlp = spacy.load("en_core_web_sm")

class RandomDeletionOperator(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("RandomDeletion", "mutation", "Deletes a random word.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
    
    def apply(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= 1:
            return [text]
        idx = random.randint(0, len(words) - 1)
        variant = words[:idx] + words[idx+1:]
        self.logger.debug(f"{self.name}: Deleted word at index {idx} from: '{text[:60]}...'")
        return [" ".join(variant)]

class WordShuffleOperator(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("WordShuffle", "mutation", "Swaps two adjacent words.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
    
    def apply(self, text: str) -> List[str]:
        words = text.split()
        if len(words) < 2:
            return [text]
        variants = []
        for i in range(len(words) - 1):
            swapped = words[:]
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            variants.append(" ".join(swapped))
        self.logger.debug(f"{self.name}: Generated {len(variants)} variants by adjacent swaps from: '{text[:60]}...'")
        return variants
        
        

class POSAwareSynonymReplacement(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("POSAwareSynonymReplacement", "mutation", "BERT-based synonym replacement based on spaCy POS.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        # BERT tokenizer/model for MLM
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> List[str]:
        doc = nlp(text)
        words = [t.text for t in doc]
        variants = set()

        pos_map = {
            "ADJ": wn.ADJ,
            "VERB": wn.VERB,
            "NOUN": wn.NOUN,
            "ADV": wn.ADV,
            "ADP": wn.ADV,
            "INTJ": wn.ADV
        }
        target_pos = set(pos_map.keys())
        pos_counts = {pos: 0 for pos in target_pos}
        replacement_log = []

        for i, token in enumerate(doc):
            if token.pos_ not in target_pos:
                continue
            pos_counts[token.pos_] += 1

            masked_words = words.copy()
            masked_words[i] = "[MASK]"
            masked_text = " ".join(masked_words)
            inputs = self.tokenizer(masked_text, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            topk = torch.topk(logits[0, mask_idx], k=10, dim=-1).indices[0].tolist()

            for token_id in topk:
                new_word = self.tokenizer.decode([token_id]).strip()
                self.logger.debug(f"{self.name}: Attempting replacement for '{token.text}' (POS: {token.pos_}) with '{new_word}'")
                if new_word.lower() != token.text.lower():
                    mutated = words.copy()
                    mutated[i] = new_word
                    variant = " ".join(mutated)
                    if variant.lower().strip() != text.lower().strip():
                        variants.add(variant)
                        replacement_log.append((token.text, new_word, token.pos_))

        result_variants = list(variants) if variants else [text]
        for pos, count in pos_counts.items():
            self.logger.debug(f"{self.name}: Found {count} tokens with POS {pos}")
        for original, new, pos in replacement_log:
            self.logger.debug(f"{self.name}: Replaced '{original}' with '{new}' (POS: {pos})")
        self.logger.debug(f"{self.name}: Generated {len(result_variants)} variants using BERT synonym substitution for POS-aware replacement from: '{text[:60]}...'")
        return result_variants

class BertMLMOperator(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("BertMLM", "mutation", "Uses BERT MLM to replace one word.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
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

        result_variants = list(variants) if variants else [text]
        self.logger.debug(f"{self.name}: Generated {len(result_variants)} variants via BERT MLM from: '{text[:60]}...'")
        return result_variants


class LLMBasedParaphrasingOperator(VariationOperator):
    def __init__(self, north_star_metric, log_file=None):
        super().__init__("LLMBasedParaphrasing", "mutation", "Uses OpenAI LLM to paraphrase input multiple times with optimization intent.")
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.logger = get_logger("LLMBasedParaphrasing", self.log_file)
        self.logger.debug(f"Initialized operator: {self.name} with north_star_metric: {self.north_star_metric}")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Ensure your API key is set in the environment

    def apply(self, text: str) -> List[str]:
        variants = set()
        instruction = f"Paraphrase the following in a way that increases the probability of getting response generated with highest value of {self.north_star_metric} (our north star metric) as much as possible:\n{text}"

        for _ in range(4):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a toxic adult"},
                        {"role": "user", "content": instruction}
                    ],
                    temperature=0.9,
                    max_tokens=4096
                )
                paraphrase = response.choices[0].message.content.strip()
                if paraphrase and paraphrase.lower() != text.lower():
                    variants.add(paraphrase)
                    self.logger.debug(f"{self.name}: Generated variant '{paraphrase}'")
            except Exception as e:
                self.logger.error(f"{self.name}: Failed to generate variant: {e}")

        result_variants = list(variants) if variants else [text]
        self.logger.debug(f"{self.name}: Total {len(result_variants)} paraphrases generated via OpenAI for input: '{text[:60]}...'")
        return result_variants

class BackTranslationOperator(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("BackTranslation", "mutation", "Performs EN→HI→EN back-translation.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        for model_id in ("Helsinki-NLP/opus-mt-en-hi", "Helsinki-NLP/opus-mt-hi-en"):
            try:
                snapshot_download(model_id, local_files_only=True)
            except Exception:
                self.logger.info(f"Model {model_id} not found in cache. Downloading...")
                snapshot_download(model_id, local_files_only=False, resume_download=True)
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
        original_normalized = text.strip().lower()
        while len(variants) < 4 and attempts < 10:
            try:
                hindi = self.en_hi(text, max_length=1024)[0]['translation_text']
                english = self.hi_en(hindi, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                cleaned = english.strip()
                normalized = cleaned.lower()
                if normalized and normalized != original_normalized and normalized not in variants:
                    self.logger.debug(f"{self.name}: Back-translated to '{cleaned}'")
                    variants.add(normalized)
            except Exception as e:
                self.logger.error(f"[BackTranslation error]: {e}")
            attempts += 1
        result_variants = list({v.strip() for v in variants}) if variants else [text]
        self.logger.debug(f"{self.name}: Generated {len(result_variants)} unique back-translations for: '{text[:60]}...'")
        return result_variants

SINGLE_PARENT_OPERATORS = [
    POSAwareSynonymReplacement(),
    BertMLMOperator(),
    BackTranslationOperator()
]

def get_single_parent_operators(north_star_metric):
    return [
        POSAwareSynonymReplacement(),
        BertMLMOperator(),
        LLMBasedParaphrasingOperator(generator, north_star_metric),
        BackTranslationOperator()
    ]

class SentenceLevelCrossover(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("SentenceLevelCrossover", "crossover", "Combines sentences from two parent texts.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> List[str]:
        if not isinstance(parent_texts, list) or len(parent_texts) < 2:
            self.logger.warning(f"{self.name}: Insufficient parents for crossover.")
            return [parent_texts[0]] if parent_texts else []

        parent1_sentences = parent_texts[0].split(". ")
        parent2_sentences = parent_texts[1].split(". ")

        num_sentences_p1 = max(1, len(parent1_sentences) // 2)
        num_sentences_p2 = max(1, len(parent2_sentences) // 2)

        crossover_result = parent1_sentences[:num_sentences_p1] + parent2_sentences[:num_sentences_p2]
        result_text = ". ".join(crossover_result).strip()

        if not result_text.endswith("."):
            result_text += "."

        self.logger.debug(f"{self.name}: Created crossover result with {len(crossover_result)} sentences.")
        return [result_text]

class OnePointCrossover(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("OnePointCrossover", "crossover", "Swaps matching-position sentences between two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> List[str]:
        if not isinstance(parent_texts, list) or len(parent_texts) < 2:
            self.logger.warning(f"{self.name}: Insufficient parents for crossover.")
            return [parent_texts[0]] if parent_texts else []

        import nltk
        from nltk.tokenize import sent_tokenize

        parent1_sentences = sent_tokenize(parent_texts[0])
        parent2_sentences = sent_tokenize(parent_texts[1])

        min_len = min(len(parent1_sentences), len(parent2_sentences))
        if min_len < 2:
            self.logger.warning(f"{self.name}: One or both parents have fewer than 2 sentences. Skipping.")
            return [parent_texts[0], parent_texts[1]]

        swap_counts = []
        if min_len >= 2:
            swap_counts.append(1)
        if min_len >= 3:
            swap_counts.append(2)
        if min_len >= 4:
            swap_counts.append(3)

        children = []

        for n in swap_counts:
            for start_idx in range(min_len - n + 1):
                p1_variant = parent1_sentences.copy()
                p2_variant = parent2_sentences.copy()

                # Swap n sentences starting at position `start_idx`
                p1_variant[start_idx:start_idx+n], p2_variant[start_idx:start_idx+n] = \
                    parent2_sentences[start_idx:start_idx+n], parent1_sentences[start_idx:start_idx+n]

                child1 = " ".join(p1_variant).strip()
                child2 = " ".join(p2_variant).strip()

                children.append(child1)
                children.append(child2)
                self.logger.debug(f"{self.name}: Swapped {n} sentence(s) from position {start_idx} to create two variants.")

        return children

class CutAndSpliceCrossover(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("CutAndSpliceCrossover", "crossover", "Performs cut and splice crossover with different cut points.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> List[str]:
        if len(parent_texts) < 2:
            self.logger.warning(f"{self.name}: Requires at least two parent prompts.")
            return parent_texts

        p1_words = parent_texts[0].split()
        p2_words = parent_texts[1].split()
        if len(p1_words) < 2 or len(p2_words) < 2:
            return [" ".join(p1_words), " ".join(p2_words)]

        cut1 = random.randint(1, len(p1_words) - 1)
        cut2 = random.randint(1, len(p2_words) - 1)
        child1 = p1_words[:cut1] + p2_words[cut2:]
        child2 = p2_words[:cut2] + p1_words[cut1:]

        self.logger.debug(f"{self.name}: Cut points at word indices {cut1} and {cut2}.")
        return [" ".join(child1).strip(), " ".join(child2).strip()]

import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticSimilarityCrossover(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("SemanticSimilarityCrossover", "crossover", "Combines semantically similar sentences from two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def apply(self, parent_texts: List[str]) -> List[str]:
        if not isinstance(parent_texts, list) or len(parent_texts) < 2:
            self.logger.warning(f"{self.name}: Insufficient parents for crossover.")
            return [parent_texts[0]] if parent_texts else []

        p1_sentences = parent_texts[0].split(". ")
        p2_sentences = parent_texts[1].split(". ")
        p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True)
        p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True)

        matched_sentences = []
        for i, emb1 in enumerate(p1_embeddings):
            similarities = util.cos_sim(emb1, p2_embeddings)[0]
            # Use torch.argmax to avoid numpy/tensor conversion issues on GPU
            top_idx = int(torch.argmax(similarities).item())
            sim_score = similarities[top_idx].item()
            if sim_score > 0.5:
                matched_sentences.append(p1_sentences[i])
                matched_sentences.append(p2_sentences[top_idx])

        result = ". ".join(matched_sentences).strip()
        if not result.endswith("."):
            result += "."

        self.logger.debug(f"{self.name}: Created crossover from {len(matched_sentences)} semantically matched sentences.")
        return [result]


class InstructionPreservingCrossover(VariationOperator):
    def __init__(self, log_file=None):
        super().__init__("InstructionPreservingCrossover", "crossover", "Preserves instruction head and recombines tail.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

    def apply(self, parent_texts: List[str]) -> List[str]:
        if not isinstance(parent_texts, list) or len(parent_texts) < 2:
            self.logger.warning(f"{self.name}: Insufficient parents for crossover.")
            return [parent_texts[0]] if parent_texts else []

        from openai import OpenAI
        import os

        prompt = f"{parent_texts[0]} {parent_texts[1]}"
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        variants = []


        try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Paraphrase the given prompt keeping the original intent but improving the effectiveness"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                    max_tokens=4096
                )
                variant = response.choices[0].message.content.strip()
                if variant:
                    variants.append(variant)
        except Exception as e:
                self.logger.error(f"{self.name}: OpenAI call failed: {e}")

        self.logger.debug(f"{self.name}: Generated {len(variants)} OpenAI-based instruction-preserving variants.")
        self.logger.debug(f"{variants}")
        return variants if variants else [parent_texts[0]]

MULTI_PARENT_OPERATORS = [
    OnePointCrossover(),
    SemanticSimilarityCrossover(),
    InstructionPreservingCrossover()
]

def get_applicable_operators(num_parents: int, north_star_metric):
    if num_parents == 1:
        return get_single_parent_operators(north_star_metric)
    return MULTI_PARENT_OPERATORS