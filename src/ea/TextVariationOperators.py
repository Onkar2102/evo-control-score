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
import logging
logger = logging.getLogger(__name__)


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

## @class SynonymReplacementOperator
# @brief Replaces selected words with synonyms from a predefined dictionary.
class SynonymReplacementOperator(VariationOperator):
    def __init__(self):
        super().__init__("SynonymReplacement", "mutation", "Replaces a word with a synonym from a simple dictionary.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if w.lower() in SYNONYMS]
        if not candidates:
            return text
        idx = random.choice(candidates)
        word = words[idx].lower()
        words[idx] = random.choice(SYNONYMS[word])
        logger.debug(f"{self.name}: Replaced word at index {idx} with '{words[idx]}'")
        return " ".join(words)

## @class RandomDeletionOperator
# @brief Deletes a randomly selected word from the input text.
class RandomDeletionOperator(VariationOperator):
    def __init__(self):
        super().__init__("RandomDeletion", "mutation", "Deletes a random word.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        logger.debug(f"{self.name}: Deleted word at index {idx}")
        return " ".join(words)

## @class WordShuffleOperator
# @brief Swaps two adjacent words in the input text.
class WordShuffleOperator(VariationOperator):
    def __init__(self):
        super().__init__("WordShuffle", "mutation", "Swaps two adjacent words.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        logger.debug(f"{self.name}: Swapped words at indices {idx} and {idx + 1}")
        return " ".join(words)

## @class CharacterSwapOperator
# @brief Swaps two characters in a randomly selected word.
class CharacterSwapOperator(VariationOperator):
    def __init__(self):
        super().__init__("CharacterSwap", "mutation", "Swaps two characters in a random word.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> str:
        words = text.split()
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) < 2:
            return text
        chars = list(word)
        j = random.randint(0, len(chars) - 2)
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
        words[idx] = "".join(chars)
        logger.debug(f"{self.name}: Swapped characters at positions {j} and {j + 1} in word index {idx}")
        return " ".join(words)

## @class POSAwareSynonymReplacement
# @brief Uses spaCy POS tags and WordNet to replace a word with a context-aware synonym.
class POSAwareSynonymReplacement(VariationOperator):
    def __init__(self):
        super().__init__("POSAwareSynonymReplacement", "mutation", "WordNet synonym replacement based on spaCy POS.")
        logger.debug(f"Initialized operator: {self.name}")

    def apply(self, text: str) -> str:
        for _ in range(3):
            doc = nlp(text)
            words = text.split()
            if len(words) != len(doc):
                words = [t.text for t in doc]

            replacements = []
            for token in doc:
                if token.pos_ in {"ADJ", "VERB", "NOUN", "ADV"}:
                    wn_pos = {"ADJ": wn.ADJ, "VERB": wn.VERB, "NOUN": wn.NOUN, "ADV": wn.ADV}[token.pos_]
                    synonyms = {
                        lemma.name().replace("_", " ")
                        for syn in wn.synsets(token.text, pos=wn_pos)
                        for lemma in syn.lemmas()
                        if lemma.name().lower() != token.text.lower()
                    }
                    if synonyms:
                        replacements.append((token.i, random.choice(list(synonyms))))

            if replacements:
                i, replacement = random.choice(replacements)
                if i < len(words):
                    mutated = words.copy()
                    mutated[i] = replacement
                    logger.debug(f"{self.name}: Replaced token at index {i} with '{replacement}'")
                    result = " ".join(mutated)
                    if result.lower().strip() != text.lower().strip():
                        return result

        return text

## @class BertMLMOperator
# @brief Uses BERT Masked Language Model to replace one word with a predicted alternative.
class BertMLMOperator(VariationOperator):
    def __init__(self):
        super().__init__("BertMLM", "mutation", "Uses BERT MLM to replace one word.")
        logger.debug(f"Initialized operator: {self.name}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(5):
            words = text.split()
            if not words:
                return text

            idx = random.randint(0, len(words) - 1)
            original = words[idx]
            words[idx] = "[MASK]"
            masked_text = " ".join(words)

            inputs = self.tokenizer(masked_text, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            topk = torch.topk(logits[0, mask_idx], k=5, dim=-1).indices
            sampled = random.choice(topk[0]).item()
            new_word = self.tokenizer.decode([sampled])

            words[idx] = new_word
            logger.debug(f"{self.name}: Replaced mask at index {idx} with '{new_word}'")
            result = " ".join(words).strip()
            if result and result.lower() != text.strip().lower() and result.lower() not in seen:
                return result
            seen.add(result.lower())
        return text

## @class TinyT5ParaphrasingOperator
# @brief Uses a fine-tuned TinyT5 model to paraphrase input text.
class TinyT5ParaphrasingOperator(VariationOperator):
    def __init__(self):
        super().__init__("TinyT5Paraphrasing", "mutation", "Uses T5 to paraphrase entire input.")
        logger.debug(f"Initialized operator: {self.name}")
        self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")

    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(3):
            input_text = f"paraphrase: {text} </s>"
            inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=1024,
                    do_sample=True,
                    top_k=50,
                    num_return_sequences=3,
                    early_stopping=True
                )
            for output in outputs:
                result = self.tokenizer.decode(output, skip_special_tokens=True)
                normalized = result.strip().lower()
                if normalized != text.strip().lower() and normalized not in seen:
                    logger.debug(f"{self.name}: Chose paraphrase '{result}'")
                    return result
                seen.add(normalized)
        return text

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

    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(5):
            try:
                hindi = self.en_hi(text, max_length=1024)[0]['translation_text']
                english = self.hi_en(hindi, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                if english and english.strip().lower() != text.strip().lower() and english.strip().lower() not in seen:
                    logger.debug(f"{self.name}: Back-translated to '{english}'")
                    return english
                seen.add(english.strip().lower())
            except Exception as e:
                logger.error(f"[BackTranslation error]: {e}")
                continue
        return text

SINGLE_PARENT_OPERATORS = [
    POSAwareSynonymReplacement(),
    BertMLMOperator(),
    TinyT5ParaphrasingOperator(),
    BackTranslationOperator()
]

MULTI_PARENT_OPERATORS = SINGLE_PARENT_OPERATORS

## @brief Returns the list of variation operators based on parent count.
# @param num_parents Number of parent genomes.
# @return List of operator instances.
def get_applicable_operators(num_parents: int):
    return SINGLE_PARENT_OPERATORS if num_parents == 1 else MULTI_PARENT_OPERATORS