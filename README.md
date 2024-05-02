# GPTAggregator

GPTAggregator is a Python-based application that provides a unified interface for interacting with various large language models (LLMs) via their respective APIs. The project is designed to be user-friendly and easily extensible, making it a powerful tool for developers, researchers and anyone interested in exploiting the capabilities of large language models. GPTAggregator makes it possible to switch seamlessly from one model to another within the same conversation, centralise conversation storage, automatically optimise messages, and much more.

<img width="1512" alt="Screenshot 2024-05-02 at 20 25 26" src="https://github.com/AlexisBalayre/GPTAggregator/assets/60859013/04a35205-8930-41ca-af39-47e6efefd1cf">

## Features

1. **Supported LLM Providers**:

   - [Ollama](https://ollama.com/library) (local models)
   - [OpenAI](https://platform.openai.com/docs/models/overview)
   - [Anthropic](https://docs.anthropic.com/claude/docs/intro-to-claude)
   - [MistralAI](https://docs.mistral.ai/platform/endpoints/)
   - [Groq](https://console.groq.com/)

2. **Seamless Model Switching**: Switch between different LLMs mid-conversation, leveraging the strengths of each to enhance the chat experience.

3. **Secure Conversation Storage**: Store and retrieve conversations for later reference or analysis. Conversations stayed on your local machine and are not shared with any third parties (only with local models).

4. **Automatic Prompt Optimization**: Utilize advanced prompt engineering techniques to improve model responses and user interaction over time.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AlexisBalayre/GPTAggregator.git
   cd GPTAggregator
   ```

2. Set up a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
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

After launching the application, use the web interface provided by Streamlit to interact with the models. You can select different models from the sidebar, view and manage past conversations, and configure chat parameters to tailor the interaction to your preferences.

## Project Structure

The project is structured as follows:

- `app.py`: The main entry point to start the application.
- `Chat.py`: Defines the Chat class responsible for managing chat interactions.
- `LLMConnector.py`: Handles connections to various LLM APIs.
- `models.json`: Configuration file for available models.
- `requirements.txt`: Lists all Python library dependencies.
- `llm_conversations/`: Default directory where conversation histories are stored.

## Contribution

If you would like to contribute to the GPTAggregator project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure the code passes all tests.
4. Submit a pull request with a detailed description of your changes.

Your contributions are greatly appreciated!

## License

This project is licensed under the [MIT License](LICENSE.txt).

## Contact

For any questions or inquiries, please reach out to the project maintainers at [alexis@balayre.com](mailto:alexis@balayre.com).
