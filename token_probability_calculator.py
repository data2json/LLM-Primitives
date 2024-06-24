#!/usr/bin/env python
# pip install requests
import json
import math
import requests


def make_api_call(question, options):
    url = "http://0.0.0.0:8000/v1/chat/completions"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {
        "messages": [
            {
                "role": "system",
                "content": f"### Instructions:\nQuestion: {question}\n\nPlease select the most appropriate response by typing only the corresponding number (1, 2, 3, 4, or 5) in your 
answer. Do not include any additional text or explanation.\n\n"
                + "\n".join(f"{i+1}. {option}" for i, option in enumerate(options)),
            }
        ],
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0,
        "max_tokens": 1,
        "top_p": 0.1,
        "n": 1,
        "logprobs": True,
        "top_logprobs": 5,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def calculate_probabilities(logprobs):
    total = sum(math.exp(lp["logprob"]) for lp in logprobs)
    return [(lp["token"], math.exp(lp["logprob"]) / total) for lp in logprobs]


def display_results(question, options, selected, probabilities):
    print(f"Question: {question}\n")
    print("Options:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"\nSelected option: {selected}\n")
    print("Token | Probability | Option")
    print("-" * 70)

    # Create a mapping of tokens to options
    token_to_option = {str(i + 1): option for i, option in enumerate(options)}

    # Sort probabilities by token to ensure they match the original order
    sorted_probs = sorted(probabilities, key=lambda x: int(x[0]))

    for token, prob in sorted_probs:
        option = token_to_option.get(token, "Unknown option")
        print(f"{token:5} | {prob:11.2%} | {option}")


def main():
    question = "You're the captain of a spaceship on a long-term mission. Your ship encounters an uncharted planet emitting strange energy signals. What's your next course of action?"
    options = [
        "Land on the planet immediately to investigate the source of the signals.",
        "Send an unmanned probe to gather more data before making a decision.",
        "Contact Earth's mission control for guidance before proceeding.",
        "Ignore the planet and continue with the original mission parameters.",
        "Conduct a thorough scan of the planet from orbit before deciding on further action.",
    ]

    api_response = make_api_call(question, options)

    selected = api_response["choices"][0]["message"]["content"]
    top_logprobs = api_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    probabilities = calculate_probabilities(top_logprobs)

    display_results(question, options, selected, probabilities)


if __name__ == "__main__":
    main()
