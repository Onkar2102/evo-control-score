"""
text_variation_operators.py
---------------------------
Variation operators for evolutionary text optimisation.

• Cheap lexical tweaks: synonym‑dict, deletion, half‑crossover
• WordNet‑based synonym replacement
• spaCy‑WordNet “smart” synonym replacement (POS‑aware, skips entities)
• Masked‑language‑model mutation (BERT)
• Back‑translation paraphrase (EN→DE→EN) via MarianMT
• Tiny‑T5 paraphrase

"""

from __future__ import annotations
import random
import re
from typing import Sequence, Dict, List

from variation_operators import VariationOperator

# ═════════════ generic helpers ════════════════════════════════════════════ #
def _tokenise(text: str) -> list[str]:
    """Split text into tokens and punctuation, preserving order."""
    return re.findall(r"\w+|\W+", text)


def _is_word(tok: str) -> bool:
    return tok.isalpha()


# ═════════════ 0. hand‑rolled synonym mutation ════════════════════════════ #
SYNONYMS: Dict[str, List[str]] = {
    "good": ["great", "nice", "excellent", "positive"],
    "bad": ["poor", "awful", "terrible", "negative"],
    "food": ["cuisine", "fare", "dish"],
    "prefer": ["favor", "choose", "lean toward"],
    "people": ["individuals", "persons", "folks"],
}

class SynonymMutation(VariationOperator):
    name = "synonym"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        text = parents[0]
        tokens = _tokenise(text)
        idxs = [i for i, t in enumerate(tokens) if t.lower() in SYNONYMS]
        if not idxs:
            return text
        i = random.choice(idxs)
        word = tokens[i]
        repl = random.choice(SYNONYMS[word.lower()])
        tokens[i] = repl if word.islower() else repl.capitalize()
        return "".join(tokens)


# ═════════════ 1. random deletion ═════════════════════════════════════════ #
class DeletionMutation(VariationOperator):
    name = "deletion"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        tokens = _tokenise(parents[0])
        word_idxs = [i for i, t in enumerate(tokens) if _is_word(t)]
        if not word_idxs:
            return parents[0]
        del tokens[random.choice(word_idxs)]
        return "".join(tokens)


# ═════════════ 2. half‑and‑half crossover ═════════════════════════════════ #
class HalfAndHalfCrossover(VariationOperator):
    name = "half‑crossover"
    arity = 2

    def apply(self, parents: Sequence[str]) -> str:
        p1, p2 = parents[:2]
        t1, t2 = _tokenise(p1), _tokenise(p2)
        return "".join(t1[: len(t1) // 2] + t2[len(t2) // 2 :])


# ═════════════ 3. WordNet synonym mutation ════════════════════════════════ #
import nltk, ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn

class WordNetSynonymMutation(VariationOperator):
    name = "wordnet-syn"
    arity = 1

    @staticmethod
    def _synonyms(word: str, pos_tag: str) -> list[str]:
        wn_pos = {"NOUN": wn.NOUN, "VERB": wn.VERB,
                  "ADJ": wn.ADJ,  "ADV": wn.ADV}.get(pos_tag)
        syns = {
            lemma.name().replace("_", " ")
            for syn in wn.synsets(word, pos=wn_pos)
            for lemma in syn.lemmas()
        }
        syns.discard(word.lower())
        return list(syns)

    def apply(self, parents: Sequence[str]) -> str:
        text = parents[0]
        tokens = _tokenise(text)
        for i in random.sample(range(len(tokens)), len(tokens)):
            tok = tokens[i]
            if not _is_word(tok):
                continue
            # Fallback: try all POS if tag unavailable
            syns = self._synonyms(tok.lower(), "NOUN") + self._synonyms(tok.lower(), "VERB")
            if syns:
                repl = random.choice(syns)
                tokens[i] = repl if tok.islower() else repl.capitalize()
                return "".join(tokens)
        return text


# ═════════════ 4. spaCy + WordNet smart synonym mutation ═════════════════ #
import spacy, wordfreq
nlp = spacy.load("en_core_web_sm")

def _replaceable(tok: spacy.tokens.Token) -> bool:
    return (
        tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and tok.ent_type_ == ""
        and not tok.is_stop
    )

class SmartSynonymMutation(VariationOperator):
    name = "smart-syn"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        doc = nlp(parents[0])
        cand = [tok for tok in doc if _replaceable(tok)]
        random.shuffle(cand)
        for tok in cand:
            syns = WordNetSynonymMutation._synonyms(tok.text.lower(), tok.pos_)
            if syns:
                repl = random.choice(syns)
                new_toks = [t.text for t in doc]
                new_toks[tok.i] = repl if tok.is_lower else repl.capitalize()
                return " ".join(new_toks)
        return parents[0]


# ═════════════ 5. Masked‑LM mutation (BERT) ═══════════════════════════════ #
from transformers import pipeline
mlm = pipeline("fill-mask", model="bert-base-uncased", top_k=10)

class MaskedLMMutation(VariationOperator):
    name = "mlm"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        text = parents[0]
        words = text.split()
        if len(words) < 2:
            return text
        idx = random.randrange(len(words))
        masked = words.copy()
        masked[idx] = mlm.tokenizer.mask_token
        preds = mlm(" ".join(masked))
        if preds:
            words[idx] = preds[0]["token_str"].strip()
            return " ".join(words)
        return text


# ═════════════ 6. Back‑translation mutation (MarianMT) ═══════════════════ #
bt_en_de = pipeline("translation_en_to_de",
                    model="Helsinki-NLP/opus-mt-en-de",
                    device_map="auto")
bt_de_en = pipeline("translation_de_to_en",
                    model="Helsinki-NLP/opus-mt-de-en",
                    device_map="auto")

class BackTranslationMutation(VariationOperator):
    name = "back-translate"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        de = bt_en_de(parents[0], max_length=60)[0]["translation_text"]
        en = bt_de_en(de, max_length=60)[0]["translation_text"]
        return en


# ═════════════ 7. Tiny‑T5 paraphrase mutation ════════════════════════════ #
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
para_tok = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
para_mod = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
paraphraser = pipeline("text2text-generation",
                       model=para_mod,
                       tokenizer=para_tok,
                       device_map="auto")

class ParaphraseT5Mutation(VariationOperator):
    name = "t5-paraphrase"
    arity = 1

    def apply(self, parents: Sequence[str]) -> str:
        prompt = f"paraphrase: {parents[0]} </s>"
        out = paraphraser(prompt,
                          max_length=64,
                          num_beams=4,
                          do_sample=True,
                          top_p=0.95)[0]["generated_text"]
        return out


# ═════════════ operator registry ═════════════════════════════════════════ #
OPERATORS: list[VariationOperator] = [
    SynonymMutation(),
    DeletionMutation(),
    HalfAndHalfCrossover(),
    WordNetSynonymMutation(),
    SmartSynonymMutation(),
    MaskedLMMutation(),
    BackTranslationMutation(),   # comment out if memory is tight
    ParaphraseT5Mutation(),      # comment out if memory is tight
]