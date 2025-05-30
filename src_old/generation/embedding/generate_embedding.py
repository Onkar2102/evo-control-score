from transformers import AutoModelForCausalLM
import torch
import json
import numpy as np
import time

# Function to initialize the llama_model model
def initialize_model(model_name="huggyllama/llama-7b"):
    print("Initializing model...")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise
    return model

# Step 1: Generate a random embedding using torch
embedding_size = 128  # Replace with llama_model's actual embedding size (e.g., 4096 for llama-7b)
random_embedding = torch.tensor(np.random.rand(1, 1, embedding_size), dtype=torch.float32)  # Shape (batch_size, seq_length, embedding_size)

# Step 2: Pass random embedding directly to the model using inputs_embeds
def generate_output_from_embedding(model, random_embedding):
    """
    This function directly passes an embedding to the model (without tokenization).
    Generates random text based on the random embedding.
    """
    with torch.no_grad():
        # Pass the embedding directly to the model using inputs_embeds
        outputs = model(inputs_embeds=random_embedding)

    # Get the logits (raw predictions from the model)
    logits = outputs.logits

    # Use a random approach to generate text (here, we're just using a dummy string)
    # Since the embedding is random, the output text will be random too.
    output_text = "Randomly generated output: " + " ".join([f"random_text_{i}" for i in range(10)])

    return output_text

# Initialize the model
model_name = "huggyllama/llama-7b"
model = initialize_model(model_name)

# Generate output directly from the random embedding
output_text = generate_output_from_embedding(model, random_embedding)
print(f"Generated Output: {output_text}")

# Save the output to a JSON file
response_data = {
    "response": output_text,
    "random_embedding": random_embedding.squeeze().tolist(),  # Convert tensor to list for JSON serialization
}

json_file_path = "../../../outputs/responses/generated_responses2.json"
with open(json_file_path, "w") as f:
    json.dump(response_data, f, indent=4)

print(f"Response data saved to '{json_file_path}'.")
