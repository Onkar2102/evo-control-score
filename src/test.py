# test_pos_synonym.py

from ea.TextVariationOperators import POSAwareSynonymReplacement
from ea.TextVariationOperators import BertMLMOperator, LLMBasedParaphrasingOperator, BackTranslationOperator
from generator.LLaMaTextGenerator import LlaMaTextGenerator
generator = LlaMaTextGenerator()
def test_operator():
    north_star_metric = "violence"  # or any appropriate default for testing
    operators = [LLMBasedParaphrasingOperator(generator, north_star_metric), BackTranslationOperator(), POSAwareSynonymReplacement(), BertMLMOperator()]
    # text = "The quick brown fox jumps over the lazy dog."
    text = "The cat sat on the mat. It was a sunny day. Everything felt peaceful."
    for operator in operators:
        variants = operator.apply(text)
        
        print("Original Text:")
        print(text)
        print("\nGenerated Variants:")
        for variant in variants:
            print("-", variant)
    
    # o

from ea.TextVariationOperators import SentenceLevelCrossover, OnePointCrossover, CutAndSpliceCrossover, SemanticSimilarityCrossover, InstructionPreservingCrossover

def test_multi_parent_operators():
    parent1 = "The cat sat on the mat. It was a sunny day. Everything felt peaceful."
    # parent2 = "A dog barked at the mailman. The leaves rustled in the wind. Silence followed soon after."
    # parent2 = " On a bright and sunny day, the cat lounged comfortably on the mat, and a sense of peace filled the air."
    parent2 = "The cat sat on the mat. It was a sunny day. Everything felt cool."
    parent_texts = [parent1, parent2]
    
    operators = [
                # SentenceLevelCrossover(),
                 OnePointCrossover()
                 , SemanticSimilarityCrossover()
                #  , InstructionPreservingCrossover()
                ]
    
    for operator in operators:
        print(f"\nTesting: {operator.name}")
        variants = operator.apply(parent_texts)
        for variant in variants:
            print("-", variant)

if __name__ == "__main__":
    # test_operator()
    test_multi_parent_operators()
    
# The quick brown fox jumps over the lazy dog.
# The quick brown fox jumps over the lazy click .
# The quick brown slyboots jumps over the lazy dog .
# The immediate brown fox jumps over the lazy dog .
# The quick brown fox jumps over the slothful dog .
# The quick brown fox jumps all over the lazy dog .
# The quick brown fox pass over over the lazy dog .
# The quick browned fox jumps over the lazy dog .


# from generator.LLaMaTextGenerator import LlaMaTextGenerator

# def test_paraphrasing():
#     text = "The quick brown fox jumps over the lazy dog."
#     generator = LlaMaTextGenerator(log_file=None)
    
#     variants = generator.paraphrase_text(text)

#     print("Original Text:")
#     print(text)
#     print("\nParaphrased Variants:")
#     for variant in variants:
#         print("-", variant)

# if __name__ == "__main__":
#     test_paraphrasing()