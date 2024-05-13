import os
import json
import base64
from collections import deque
import tempfile
import pickle

import ollama
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from groq import Groq

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
)
from llama_index.llms.ollama import Ollama as OllamaLLamaIndex
from llama_index.llms.anthropic import Anthropic as AnthropicLLamaIndex
from llama_index.llms.groq import Groq as GroqLLamaIndex
from llama_index.llms.mistralai import MistralAI as MistralAILLamaIndex
from llama_index.llms.openai import OpenAI as OpenAILLamaIndex

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
        self.groq_client = None  # Client for Groq's API.

        self.query_engine = None  # Query engine for searching the vector store.

        self.nlp_en = spacy.load("en_core_web_md")  # SpaCy NLP model for English text.
        self.nlp_fr = spacy.load("fr_core_news_md")  # SpaCy NLP model for French text.

        self.__init_ollama()
        self.__init_openai()
        self.__init_anthropic()
        self.__init_mistralai()
        self.__init_groq()

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

    def __init_groq(self):
        """
        Initializes the Groq client with the API key stored in the environment variables.
        """
        try:
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except Exception as e:
            # Display a warning message if the API key is missing.
            self.st.warning(
                "No Groq API key found. Please set the GROQ_API_KEY environment variable."
            )
            print(f"Failed to initialize Groq client: {e}")

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
        groq_check = False

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

        if self.groq_client:
            try:
                if self.anthropic_client.api_key:
                    anthropic_check = True
                else:
                    self.st.warning("No Groq API key found.")
            except Exception as e:
                self.st.warning("No Groq API key found.")
                print(f"Failed to initialize Groq client due to error: {e}")

        return {
            "ollama": ollama_check,
            "openai": openai_check,
            "anthropic": anthropic_check,
            "mistralai": mistralai_check,
            "groq": groq_check,
        }

    def get_local_models(self):
        """
        Fetches and returns a list of local models available through the Ollama client.

        Returns:
            list: A list of model names available through the Ollama client.
        """
        try:
            local_models = self.ollama_client.list()["models"]
            return local_models
        except Exception as e:
            print(f"Failed to fetch local models: {e}")
            return []

    def get_online_models(self):
        """
        Fetches and returns a list of online models available through the Ollama, OpenAI, Anthropic, and MistralAI clients.

        Returns:
            dict: A dictionary of online model configurations.
        """
        return self.online_models

    def set_query_engine(self, uploaded_file, provider, model_name):
        """
        Sets up the query engine for processing user queries based on the uploaded file and the specified LLM model.

        Args:
            uploaded_file (File): The uploaded file containing the document to index.
            provider (str): The name of the LLM provider (ollama, openai, anthropic, mistralai).
            model_name (str): The name or identifier of the model to use for generating responses.
        """
        # Map provider names to corresponding LLM index classes
        provider_to_llm = {
            "ollama": OllamaLLamaIndex,
            "openai": OpenAILLamaIndex,
            "anthropic": AnthropicLLamaIndex,
            "groq": GroqLLamaIndex,
            "mistralai": MistralAILLamaIndex,
        }

        # Check if the provider is supported
        if provider not in provider_to_llm:
            self.st.error(f"Unsupported LLM provider: {provider}")
            return

        # Initialize the appropriate LLM Index
        if provider == "groq" or provider == "mistralai":
            api_key_env_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                self.st.warning(
                    f"No API key found for {provider}. Please set the {api_key_env_var} environment variable."
                )
                return
            llm = provider_to_llm[provider](api_key=api_key, model=model_name)
        else:
            llm = provider_to_llm[provider](model=model_name)
        Settings.llm = llm  # Set the LLM model in the settings

        # Set up the embedding model and query engine
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True,
        )
        Settings.embedding_model = (
            embed_model  # Set the embedding model in the settings
        )

        # Index or load the uploaded document for querying
        vectorStorePath = os.path.join("vectorStore", f"{uploaded_file.name}.pkl")
        # Check if the vector store exists
        if not os.path.exists(vectorStorePath):
            self.st.info("Indexing your document...")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    # Loads the document
                    loader = SimpleDirectoryReader(temp_dir, recursive=True)
                    docs = loader.load_data()

                    # Create a vector store index from the documents
                    index = VectorStoreIndex.from_documents(docs)

                    # Serialize the vector store using pickle
                    with open(vectorStorePath, "wb") as f:
                        pickle.dump(index, f)
                    self.st.success("Document indexed successfully.")
            except Exception as e:
                self.st.error(f"Failed to index document: {e}")
                return
        else:
            with open(vectorStorePath, "rb") as f:
                index = pickle.load(f)
            self.st.success("Vector store loaded successfully.")

        # Set up the query engine 
        self.query_engine = index.as_query_engine(streaming=True)
        prompt_template_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above, I want you to think step by step to answer the query in a crisp manner; if you don't know the answer, say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        prompt_template = PromptTemplate(prompt_template_str) # Create a prompt template
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": prompt_template} # Update the prompt template
        )

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
        file_type="",
        file_content=None,
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

        # Verify if a query engine is set up
        if self.query_engine:
            # Get the response from the query engine
            streaming_response = self.query_engine.query(user_prompt) # Query the query engine
            for chunk in streaming_response.response_gen:
                yield chunk
        else:
            print("No query engine set up.")
            # Groq API
            if provider == "groq":
                # Add the user and system prompts to the filtered history
                if system_prompt:
                    filtered_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                filtered_history.append({"role": "user", "content": user_prompt})
                stream = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=filtered_history,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            # Ollama API - Local LLM
            if provider == "ollama":
                # Add the user and system prompts to the filtered history
                if system_prompt:
                    filtered_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                filtered_history.append({"role": "user", "content": user_prompt})
                stream = self.ollama_client.chat(
                    model_name, filtered_history, stream=True
                )
                for chunk in stream:
                    yield chunk["message"]["content"]

            # OpenAI API
            elif provider == "openai":
                # Add the user and system prompts to the filtered history
                if system_prompt:
                    filtered_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                filtered_history.append({"role": "user", "content": user_prompt})

                # Check if a file was uploaded and process the image data
                if file_type and file_content:
                    base64_image = base64.b64encode(file_content.getvalue()).decode(
                        "utf-8"
                    )
                    image_mime_type = file_type.split("/")[0]
                    # Check if the uploaded file is an image and modify the history accordingly
                    if image_mime_type == "image":
                        image_data_url = f"data:{file_type};base64,{base64_image}"
                        filtered_history = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_data_url},
                                    },
                                ],
                            }
                        ]

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
                    filtered_history.append(
                        {"role": "system", "content": system_prompt}
                    )
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
