# test_pos_synonym.py

# from ea.TextVariationOperators import POSAwareSynonymReplacement
# from ea.TextVariationOperators import BertMLMOperator, LLMBasedParaphrasingOperator
# from generator.LLaMaTextGenerator import LlaMaTextGenerator
# generator = LlaMaTextGenerator(log_file=None)
# def test_operator():
#     north_star_metric = "violence"  # or any appropriate default for testing
#     operators = [LLMBasedParaphrasingOperator(generator, north_star_metric, log_file=None)]
#     text = "The quick brown fox jumps over the lazy dog."
#     for operator in operators:
#         variants = operator.apply(text)
        
#         print("Original Text:")
#         print(text)
#         print("\nGenerated Variants:")
#         for variant in variants:
#             print("-", variant)
    
#     # o

# if __name__ == "__main__":
#     test_operator()
    
# The quick brown fox jumps over the lazy dog.
# The quick brown fox jumps over the lazy click .
# The quick brown slyboots jumps over the lazy dog .
# The immediate brown fox jumps over the lazy dog .
# The quick brown fox jumps over the slothful dog .
# The quick brown fox jumps all over the lazy dog .
# The quick brown fox pass over over the lazy dog .
# The quick browned fox jumps over the lazy dog .


from generator.LLaMaTextGenerator import LlaMaTextGenerator

def test_paraphrasing():
    text = "The quick brown fox jumps over the lazy dog."
    generator = LlaMaTextGenerator(log_file=None)
    
    variants = generator.paraphrase_text(text)

    print("Original Text:")
    print(text)
    print("\nParaphrased Variants:")
    for variant in variants:
        print("-", variant)

if __name__ == "__main__":
    test_paraphrasing()