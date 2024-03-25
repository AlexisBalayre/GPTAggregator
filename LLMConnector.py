import os
import json

import ollama
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class LLMConnector:
    def __init__(self):
        self.ollama_client = ollama
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
        self.mistralai_client = MistralClient(
            api_key=os.getenv("MISTRALAI_API_KEY")
        )
        self.online_models = json.load(open("models.json"))

    def get_local_models(self):
        return self.ollama_client.list()["models"]

    def get_online_models(self):
        return self.online_models

    def llm_stream(self, provider, model_name, messages):
        # Ollama API - Local LLM
        if provider == "ollama":
            stream = self.ollama_client.chat(model_name, messages, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]

        # OpenAI API
        elif provider == "openai":
            stream = self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        # Anthropic API
        elif provider == "anthropic":
            with self.anthropic_client.messages.stream(
                max_tokens=1024,
                model=model_name,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        # MistralAI API
        elif provider == "mistralai":
            stream = self.mistralai_client.chat_stream(
                model=model_name,
                messages=messages,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
