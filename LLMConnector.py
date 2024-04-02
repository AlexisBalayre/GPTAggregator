import os
import json
from collections import deque

import ollama
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
import spacy


class LLMConnector:
    """
    A connector class designed to interface with various large language models (LLMs) through their respective APIs.
    Supports local and online model querying, as well as streaming responses from these models.

    Parameters:
        st (Streamlit): The Streamlit instance to use for logging and displaying messages.
    """

    def __init__(self, st):
        """
        Initializes the LLMConnector with client instances for each supported API and loads the configuration for online models.
        """
        self.st = st

        self.ollama_client = None  # Client for interacting with local Ollama models.
        self.openai_client = None  # Client for OpenAI's API.
        self.anthropic_client = None  # Client for Anthropic's API.
        self.mistralai_client = None  # Client for MistralAI's API.

        self.nlp_en = spacy.load("en_core_web_md")  # SpaCy NLP model for English text.
        self.nlp_fr = spacy.load("fr_core_news_md")  # SpaCy NLP model for French text.

        self.__init_ollama()
        self.__init_openai()
        self.__init_anthropic()
        self.__init_mistralai()

        self.online_models = json.load(
            open("models.json")
        )  # Loads the online model configurations from a JSON file.

    def __init_ollama(self):
        """
        Initializes the Ollama client for interacting with local models.
        """
        try:
            self.ollama_client = ollama
            # Count the number of available models.
            model_count = len(self.ollama_client.list()["models"])
            if model_count == 0:
                # Display a warning message if no models are available.
                self.st.warning("No local models found.")
        except Exception as e:
            # Display a warning message if the Ollama client fails to initialize.
            self.st.warning("Failed to initialize Ollama client.")
            print(f"Failed to initialize Ollama client: {e}")

    def __init_openai(self):
        """
        Initializes the OpenAI client with the API key stored in the environment variables.
        """
        try:
            self.openai_client = OpenAI()
        except Exception as e:
            # Display a warning message if the API key is missing.
            self.st.warning(
                "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
            )
            print(f"Failed to initialize OpenAI client: {e}")

    def __init_mistralai(self):
        """
        Initializes the MistralAI client with the API key stored in the environment variables.
        """
        try:
            self.mistralai_client = MistralClient(
                api_key=os.getenv("MISTRALAI_API_KEY")
            )
        except Exception as e:
            # Display a warning message if the API key is missing.
            self.st.warning(
                "No MistralAI API key found. Please set the MISTRALAI_API_KEY environment variable."
            )
            print(f"Failed to initialize MistralAI client: {e}")

    def __init_anthropic(self):
        """
        Initializes the Anthropic client with the API key stored in the environment variables.
        """
        try:
            self.anthropic_client = Anthropic()
        except Exception as e:
            # Display a warning message if the API key is missing.
            self.st.warning(
                "No Anthropic API key found. Please set the ANTHROPIC_API_KEY environment variable."
            )
            print(f"Failed to initialize Anthropic client: {e}")

    def health_check(self):
        """
        Performs a health check on the Ollama, OpenAI, Anthropic, and MistralAI clients to verify their availability.

        Returns:
            dict: A dictionary containing the health status of each client.
        """
        print("Performing health check...")
        ollama_check = False
        openai_check = False
        anthropic_check = False
        mistralai_check = False

        if self.ollama_client:
            try:
                _ = self.ollama_client.list()["models"]
                ollama_check = True
            except Exception as e:
                print(f"Failed to initialize Ollama client due to error: {e}")

        if self.openai_client:
            try:
                _ = self.openai_client.models.list()
                openai_check = True
            except Exception as e:
                self.st.warning("No OpenAI API key found.")
                print(f"Failed to initialize OpenAI client due to error: {e}")

        if self.anthropic_client:
            try:
                if self.anthropic_client.api_key:
                    anthropic_check = True
                else:
                    self.st.warning("No Anthropic API key found.")
            except Exception as e:
                self.st.warning("No Anthropic API key found.")
                print(f"Failed to initialize Anthropic client due to error: {e}")

        if self.mistralai_client:
            try:
                _ = self.mistralai_client.list_models()
                mistralai_check = True
            except Exception as e:
                self.st.warning("No MistralAI API key found.")
                print(f"Failed to initialize MistralAI client due to error: {e}")

        return {
            "ollama": ollama_check,
            "openai": openai_check,
            "anthropic": anthropic_check,
            "mistralai": mistralai_check,
        }

    def get_local_models(self):
        """
        Fetches and returns a list of local models available through the Ollama client.

        Returns:
            list: A list of model names available through the Ollama client.
        """
        return self.ollama_client.list()["models"]

    def get_online_models(self):
        """
        Fetches and returns a list of online models available through the Ollama, OpenAI, Anthropic, and MistralAI clients.

        Returns:
            dict: A dictionary of online model configurations.
        """
        return self.online_models

    def filter_messages(
        self,
        messages_history,
        user_prompt,
        system_prompt,
        max_history_length=10,
        similarity_threshold=0.5,
        language="English",
    ):
        """
        Filters the message history based on semantic similarity and relevance to the current prompts.

        Args:
            messages_history (list): A list of message objects representing the conversation history.
            user_prompt (str): The user's input prompt for generating the next response.
            system_prompt (str): The system's input prompt for generating the next response.
            max_history_length (int): The maximum number of messages to keep in the history (default=10).
            similarity_threshold (float): The minimum similarity score required to include a message in the history (default=0.5).

        Returns:
            list: A filtered list of message objects representing the optimized conversation history.
        """
        # Convert the message history to a list of dictionaries
        history = [
            {"role": msg.get("role", ""), "content": msg.get("content", "")}
            for msg in messages_history
            if msg.get("role", "") in ["user", "assistant"]
        ]

        # Deque to store the filtered message history
        filtered_history = deque(maxlen=max_history_length * 2)

        # Select the appropriate SpaCy NLP model based on the language
        self.nlp = self.nlp_en if language.lower() == "english" else self.nlp_fr

        # Initialize the SpaCy NLP model for processing text data
        if language.lower() in ["english", "french"]:
            # Create SpaCy documents for the user and system prompts
            user_doc = self.nlp(user_prompt)
            system_doc = self.nlp(system_prompt)

            # Iterate over the message history in reverse order
            for msg in reversed(history):
                msg_doc = self.nlp(msg["content"])

                # Calculate the similarity scores between the message and the prompts
                user_similarity = user_doc.similarity(msg_doc)
                system_similarity = system_doc.similarity(msg_doc)

                # If the message is sufficiently similar to either prompt, add it to the filtered history
                if max(user_similarity, system_similarity) >= similarity_threshold:
                    filtered_history.appendleft(msg)
        else:
            # If the language is not English, simply append the last few messages to the filtered history
            filtered_history.extend(history[-(max_history_length * 2) :])

        return list(filtered_history)

    def llm_stream(
        self,
        provider,
        model_name,
        messages_history,
        user_prompt,
        system_prompt,
        language="en",
        history_length=10,
        similarity_threshold=0.5,
        max_tokens=1024,
    ):
        """
        Streams responses from the specified language model in real-time. This method handles interactions with different LLM providers by yielding messages as they become available.

        Args:
            provider (str): The name of the LLM provider (ollama, openai, anthropic, mistralai).
            model_name (str): The name or identifier of the model to use for generating responses.
            messages_history (list): A list of message objects representing the conversation history.
            user_prompt (str): The user's input prompt for generating the next response.
            system_prompt (str): The system's input prompt for generating the next response.

        Yields:
            str: A response message generated by the language model.
        """

        # Filter the message history based on semantic similarity
        filtered_history = self.filter_messages(
            messages_history=messages_history,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_history_length=history_length,
            similarity_threshold=similarity_threshold,
            language=language,
        )

        # Ollama API - Local LLM
        if provider == "ollama":
            # Add the user and system prompts to the filtered history
            if system_prompt:
                filtered_history.append({"role": "system", "content": system_prompt})
            filtered_history.append({"role": "user", "content": user_prompt})
            stream = self.ollama_client.chat(model_name, filtered_history, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]

        # OpenAI API
        elif provider == "openai":
            # Add the user and system prompts to the filtered history
            if system_prompt:
                filtered_history.append({"role": "system", "content": system_prompt})
            filtered_history.append({"role": "user", "content": user_prompt})
            stream = self.openai_client.chat.completions.create(
                model=model_name,
                messages=filtered_history,
                stream=True,
                max_tokens=max_tokens,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        # Anthropic API
        elif provider == "anthropic":
            # Add the user prompt to the filtered history
            filtered_history.append({"role": "user", "content": user_prompt})
            with self.anthropic_client.messages.stream(
                max_tokens=max_tokens,
                model=model_name,
                messages=filtered_history,
                system=system_prompt,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        # MistralAI API
        elif provider == "mistralai":
            # Add the user and system prompts to the filtered history
            if system_prompt:
                filtered_history.append({"role": "system", "content": system_prompt})
            filtered_history.append({"role": "user", "content": user_prompt})
            stream = self.mistralai_client.chat_stream(
                model=model_name,
                messages=filtered_history,
                max_tokens=max_tokens,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
