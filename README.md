# LLM-Primitives

Large Language Model Primitive Exploration and Demonstration

## Project Overview

LLM-Primitives is a project dedicated to exploring and demonstrating the fundamental building blocks (primitives) of Large Language Models (LLMs). This repository contains various tools and demonstrations that showcase different aspects of LLM functionality, helping developers and researchers better understand and utilize these powerful AI models.

## Features

- Token Probability Calculator: A tool that demonstrates how LLMs assign probabilities to different tokens when making predictions.
- [Other features to be added as the project expands]

## Technologies Used

- VLLM: A high-throughput and memory-efficient inference engine for LLMs
- LLAMA-3: The latest iteration of Meta's Large Language Model

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/data2json/LLM-Primitives.git
   cd LLM-Primitives
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Token Probability Calculator

The Token Probability Calculator is a Python script that interacts with an LLM API to showcase how the model assigns probabilities to different options when presented with a question.

To run the calculator:

1. Ensure you have access to the LLM API (default URL: `http://0.0.0.0:8000`).
2. Run the script:
   ```
   python token_probability_calculator.py
   ```

3. The script will output the question, available options, the model's selected option, and the probability distribution for each option.

## API Integration

The project currently integrates with a local LLM API. To use a different API:

1. Open `token_probability_calculator.py`
2. Modify the `url` variable in the `make_api_call` function to point to your desired API endpoint.
3. Adjust the request payload if necessary to match your API's requirements.

## Contributing

Contributions to LLM-Primitives are welcome! Please feel free to submit pull requests, create issues, or suggest new primitives to explore.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the BSD 3-part License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of VLLM and Meta AI for LLAMA-3.
- Inspired by the growing field of AI and language models.

## Contact

Essobi - [@essobi](https://twitter.com/essobi) - Essobi@gmail.com

Project Link: [https://github.com/data2json/LLM-Primitives](https://github.com/data2json/LLM-Primitives)
