import os
import json
import datetime
from uuid import uuid4

import streamlit as st
from LLMConnector import LLMConnector


class Chat:
    def __init__(self):
        self.llmConnector = LLMConnector()
        self.LOCAL_MODELS = self.llmConnector.get_local_models()
        self.ONLINE_MODELS = self.llmConnector.get_online_models()
        self.OUTPUT_DIR = "llm_conversations"
        self.st = st
        self.__configure_chat()

    def __configure_chat(self):
        self.st.set_page_config(
            layout="wide", page_title="GPTAggregator", page_icon="🚀"
        )
        self.st.sidebar.title("GPTAggregator 🚀")

        self.selected_model = self.select_model()
        self.display_conversation_history()
        self.new_conversation()

    def run(self):
        prompt = self.st.chat_input(f"Ask '{self.selected_model}' a question ...")
        self.chat(prompt)

    def new_conversation(self):
        new_conversation = self.st.sidebar.button(
            "New conversation", key="new_conversation"
        )
        if new_conversation:
            self.st.session_state["conversation_id"] = str(uuid4())
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = []
            file_name = f"{self.st.session_state['conversation_id']}_{self.selected_model[1]}.json"
            json.dump([], open(os.path.join(self.OUTPUT_DIR, file_name), "w"))
            self.st.rerun()

    def select_model(self):
        local_models_names = [
            model["name"] for model in self.LOCAL_MODELS
        ]  # Local models
        online_models_names = [
            model["name"] for model in self.ONLINE_MODELS
        ]  # Online models
        model_names = local_models_names + online_models_names  # All models

        # Sidebar - Model selection
        self.st.sidebar.subheader("Models")
        llm_name = self.st.sidebar.selectbox(
            f"Select Model (available {len(model_names)})", model_names
        )

        # Model details
        if llm_name:
            # local model
            if llm_name in local_models_names:
                llm_details = [
                    model for model in self.LOCAL_MODELS if model["name"] == llm_name
                ][0]
                if type(llm_details["size"]) != str:
                    llm_details["size"] = f"{round(llm_details['size'] / 1e9, 2)} GB"
                llm_provider = "ollama"

            # online model
            else:
                llm_details = [
                    model for model in self.ONLINE_MODELS if model["name"] == llm_name
                ][0]
                llm_provider = llm_details["provider"]
                llm_name = llm_details["modelName"]

            with self.st.expander("LLM Details"):
                self.st.write(llm_details)

            return [llm_provider, llm_name]

        return [self.LOCAL_MODELS[0]["provider"], self.LOCAL_MODELS[0]["name"]]

    def display_conversation_history(self):
        OUTPUT_DIR = os.path.join(os.getcwd(), self.OUTPUT_DIR)

        # Get a list of all conversation files for the selected model
        conversation_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]

        # Add an empty conversation file at the beginning of the list
        conversation_files.insert(0, "")

        # Sidebar - Conversation history
        self.st.sidebar.subheader("Conversation History")
        selected_conversation = self.st.sidebar.selectbox(
            "Select a conversation", conversation_files, index=0
        )

        if selected_conversation:
            # Load the conversation data from the selected file
            conversation_file = os.path.join(OUTPUT_DIR, selected_conversation)
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)

            self.st.session_state["conversation_id"] = selected_conversation.split("_")[
                0
            ]

            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = conversation_data

            # Display the conversation in the main area
            for message in conversation_data:
                role = message["role"]
                if role == "user":
                    with self.st.chat_message("user"):
                        question = message["content"]
                        self.st.markdown(f"{question}", unsafe_allow_html=True)
                elif role == "assistant":
                    with self.st.chat_message("assistant"):
                        self.st.markdown(message["content"], unsafe_allow_html=True)

    def chat(self, prompt):
        if "conversation_id" in self.st.session_state:
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
        else:
            # Create a new conversation
            self.st.session_state["conversation_id"] = str(uuid4())
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
            self.st.session_state[chat_history_key] = []

        # Display chat history
        for message in self.st.session_state[chat_history_key]:
            role = message["role"]
            if role == "user":
                with self.st.chat_message("user"):
                    question = message["content"]
                    self.st.markdown(f"{question}", unsafe_allow_html=True)
            elif role == "assistant":
                with self.st.chat_message("assistant"):
                    self.st.markdown(message["content"], unsafe_allow_html=True)

        # run the model
        if prompt:
            self.st.session_state[chat_history_key].append(
                {"content": f"{prompt}", "role": "user"}
            )
            with self.st.chat_message("user"):
                self.st.write(prompt)

            messages = [
                dict(content=message["content"], role=message["role"])
                for message in self.st.session_state[chat_history_key]
            ]

            with self.st.chat_message("assistant"):
                chat_box = self.st.empty()
                response_message = chat_box.write_stream(
                    self.llmConnector.llm_stream(
                        self.selected_model[0], self.selected_model[1], messages
                    )
                )

            self.st.session_state[chat_history_key].append(
                {"content": f"{response_message}", "role": "assistant"}
            )

            self.save_conversation()

            return response_message

    def save_conversation(self):
        conversation_id = self.st.session_state["conversation_id"]
        conversation_key = f"chat_history_{conversation_id}"
        conversation_chat = self.st.session_state[conversation_key]
        filename = f"{conversation_id}_{self.selected_model[1]}.json"
        if conversation_chat:
            conversation_file = os.path.join(self.OUTPUT_DIR, filename)
            if os.path.exists(conversation_file):
                with open(conversation_file, "r") as f:
                    existing_conversation_data = json.load(f)
                conversation_chat.extend(existing_conversation_data)
            with open(conversation_file, "w") as f:
                json.dump(conversation_chat, f, indent=4)
                self.st.success(f"Conversation saved to {conversation_file}")
