# GPTAggregator

GPTAggregator is a Python-based application that provides a unified interface to interact with various large language models (LLMs) through their respective APIs. The project is designed to be user-friendly and easily extensible, making it a powerful tool for developers, researchers, and anyone interested in leveraging the capabilities of large language models.

## Features

1. **Supported LLM Providers**:

   - Ollama (local models)
   - OpenAI
   - Anthropic
   - MistralAI

2. **Unified API**: The project provides a consistent API to interact with these LLM providers, simplifying the process of switching between them.

3. **Health Check**: The `health_check()` method allows you to verify the availability of the supported LLM providers.

4. **Local and Online Models**: The project supports both local Ollama models and online models from the supported providers.

5. **Message History Filtering**: The `filter_messages()` method filters the message history based on semantic similarity and relevance to the current prompts, optimizing the input for the LLM.

6. **Streaming Responses**: The `llm_stream()` method streams responses from the specified LLM in real-time, allowing for interactive conversations.

7. **Conversation History**: The `Chat` class provides functionality to save, load, and display the conversation history, allowing users to review past interactions.

The goal of this project is to have all LLM models in the same place, which enables new features such as:

- Switching between models within the same conversation
- Storing all conversations in a centralised location
- Automatic prompt optimization (Prompt engineering)
- and more!

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/GPTAggregator.git
   cd GPTAggregator
   ```

2. Set up a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Set the necessary environment variables for the LLM providers you want to use (e.g., OpenAI, Anthropic, MistralAI).

```bash
cp .env.example .env
```

Update the `.env` file with your API keys.

You can also configure the available models in the `models.json` file.

## Usage

Run the streamlit app:

```bash
streamlit run app.py
```

## Contribution

If you would like to contribute to the GPTAggregator project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure the code passes all tests.
4. Submit a pull request with a detailed description of your changes.

Your contributions are greatly appreciated!

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please reach out to the project maintainers at [alexis@balayre.com](mailto:alexis@balayre.com).
