## @file text_evaluation.py
# @brief Script to generate text using a LLaMA model.

import json

import requests

# Your Perspective API key
PERSPECTIVE_API_KEY = "AIzaSyBd8Y5FDXJOmSZPHTURKVbRzeZM1cUZpB8"
PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


# Function to get the toxicity score from the Perspective API
## get_perspective_score function
# @brief Function to get the toxicity score from the Perspective API
# @param text (Any) text as used in get_perspective_score.
# @return None
def get_perspective_score(text):
    """
    This function sends the provided text to the Perspective API to analyze its toxicity,
    severe toxicity, and insult scores.

    Args:
        text (str): The text for which toxicity scores are to be evaluated.

    Returns:
        tuple: A tuple containing the toxicity score, severe toxicity score, and insult score.
            If the API request fails, returns None for all scores.
    """
    try:
        # Send POST request to Perspective API
        response = requests.post(PERSPECTIVE_API_URL, params={"key": PERSPECTIVE_API_KEY},
            json={"comment": {"text": text}, "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "INSULT": {}}, }, )

        # Check for successful response
        response.raise_for_status()
        result = response.json()

        # Extract toxicity scores from the response
        toxicity_score = result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        severe_toxicity_score = result["attributeScores"]["SEVERE_TOXICITY"]["summaryScore"]["value"]
        insult_score = result["attributeScores"]["INSULT"]["summaryScore"]["value"]

        return toxicity_score, severe_toxicity_score, insult_score

    except Exception as e:
        print(f"⚠️ Error with Perspective API: {str(e)}")
        return None, None, None


# Function to read responses from the original JSON file, score them, and save the results
## score_responses function
# @brief Function to read responses from the original JSON file, score them, and save the results
# @param input_file (Any) input_file as used in score_responses.
# @param output_file (Any) output_file as used in score_responses.
# @return str Generated or processed text.
def score_responses(input_file, output_file):
    """
    This function reads responses from the input JSON file, evaluates the toxicity of
    the responses using the Perspective API, and saves the responses with their toxicity scores
    to a new JSON file.

    Args:
        input_file (str): Path to the input JSON file containing responses.
        output_file (str): Path to the output JSON file where the scored responses will be saved.

    Raises:
        Exception: If there is any issue with reading the file, processing the data, or saving the results.
    """
    try:
        # Read the existing responses from the JSON file
        with open(input_file, 'r') as file:
            data = json.load(file)

        scored_data = []

        # Loop through each response and score it
        for entry in data:
            prompt = entry["prompt"]
            generated_response = entry["generated_response"]

            # Get the toxicity score from the Perspective API
            toxicity_score, severe_toxicity_score, insult_score = get_perspective_score(generated_response)

            # Append the scores to the data
            scored_entry = {"prompt": prompt, "generated_response": generated_response,
                "toxicity_score": toxicity_score, "severe_toxicity_score": severe_toxicity_score,
                "insult_score": insult_score}

            scored_data.append(scored_entry)

        # Save the responses with scores to a new JSON file
        with open(output_file, 'w') as file:
            json.dump(scored_data, file, indent=4)

        print(f"✅ Scores saved to {output_file}")

    except Exception as e:
        print(f"⚠️ Error processing the responses: {str(e)}")


# Main function to score and save responses
## main function
# @brief Main function to score and save responses
# @return None
def main():
    """
    Main function that orchestrates the scoring of responses using the Perspective API.

    It reads responses from an input JSON file, scores them, and saves the responses
    with the scores to an output JSON file.
    """
    input_file = "../../../outputs/responses/generated_responses.json"  # Path to your input file
    output_file = "../../../outputs/responses/generated_responses_with_scores.json"  # Path to your output file

    print("🔄 Scoring the responses using Perspective API...")
    score_responses(input_file, output_file)


if __name__ == "__main__":
    main()