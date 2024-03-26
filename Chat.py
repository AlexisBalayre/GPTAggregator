import os
import json
import datetime

import streamlit as st
from LLMConnector import LLMConnector


class Chat:
    """
    A Chat class to handle interactions with various language models via a Streamlit interface.
    This class facilitates selecting models, initiating conversations, displaying conversation history,
    and managing input/output of chat messages.
    """

    def __init__(self, output_dir="llm_conversations"):
        """
        Initializes the Chat class, setting up model connections and configuring the chat interface.

        Args:
            output_dir (str): The directory to save conversation history files.
        """
        self.st = st  # Streamlit instance for UI rendering
        self.llmConnector = LLMConnector(st=self.st)  # Connect to LLM API
        self.LOCAL_MODELS = self.llmConnector.get_local_models()  # Load local models
        self.ONLINE_MODELS = self.llmConnector.get_online_models()  # Load online models
        self.OUTPUT_DIR = output_dir  # Directory for saving conversations
        self.__configure_chat()  # Configure the chat UI

    def __configure_chat(self):
        """
        Configures the Streamlit interface for the chat application, including page layout and model selection.
        """
        # Set Streamlit page config
        self.st.set_page_config(
            layout="wide", page_title="GPTAggregator", page_icon="🚀"
        )
        # Add sidebar title
        self.st.sidebar.title("GPTAggregator 🚀")
        # Select the model for conversation
        self.selected_model = self.select_model()
        # Display previous conversations
        self.display_conversation_history()
        # Provide option for new conversation
        self.new_conversation()

    def run(self):
        """
        Runs the chat interface allowing for user input and displays responses from the selected model.
        """
        # Input box for user's questions
        prompt = self.st.chat_input(f"Ask {self.selected_model[1]} a question ...")
        # Process and display the conversation
        self.chat(prompt)

    def new_conversation(self):
        """
        Initiates a new conversation, generating a unique identifier and resetting chat history.
        """
        # Button to start a new conversation
        new_conversation = self.st.sidebar.button(
            "New conversation", key="new_conversation"
        )
        if new_conversation:
            # Generate unique conversation ID
            self.st.session_state["conversation_id"] = str(datetime.datetime.now())
            # Reset chat history for the new conversation
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = []
            # Prepare file name for saving the conversation
            file_name = f"{self.st.session_state['conversation_id']}.json"
            # Initialize the conversation file with empty content
            json.dump([], open(os.path.join(self.OUTPUT_DIR, file_name), "w"))
            # Rerun the app to reflect changes
            self.st.rerun()

    def select_model(self):
        """
        Allows the user to select a language model for the conversation, providing details for both local and online models.

        Returns:
            A list containing the selected model's provider and name.
        """
        # Health check for the LLM providers
        # health_check = self.llmConnector.health_check()

        # Compile lists of available models
        local_models_names = [
            model["name"] for model in self.LOCAL_MODELS  # if health_check["ollama"]
        ]  # Local models
        online_models_names = [
            model["name"]
            for model in self.ONLINE_MODELS  # if health_check[model["provider"]]
        ]  # Online models
        model_names = local_models_names + online_models_names  # Combine all models

        # Sidebar selection for model
        self.st.sidebar.subheader("Models")
        llm_name = self.st.sidebar.selectbox(
            f"Select a model ({len(model_names)} available)", model_names
        )

        # Check if the selected model is local or online and extract its details accordingly
        if llm_name:
            # If the model is local
            if llm_name in local_models_names:
                llm_details = [
                    model for model in self.LOCAL_MODELS if model["name"] == llm_name
                ][0]
                if type(llm_details["size"]) != str:
                    llm_details["size"] = f"{round(llm_details['size'] / 1e9, 2)} GB"
                llm_provider = "ollama"

            # If the model is online
            else:
                llm_details = [
                    model for model in self.ONLINE_MODELS if model["name"] == llm_name
                ][0]
                llm_provider = llm_details["provider"]
                llm_name = llm_details["modelName"]

            # Display model details for the user's reference
            with self.st.expander("LLM Details"):
                self.st.write(llm_details)

            return [llm_provider, llm_name]

        # Return the default model if no model is selected
        return [self.LOCAL_MODELS[0]["provider"], self.LOCAL_MODELS[0]["name"]]

    def display_conversation_history(self):
        """
        Displays the conversation history for the selected model, allowing users to review past interactions.
        """
        # Define the directory where conversation history files are stored
        OUTPUT_DIR = os.path.join(os.getcwd(), self.OUTPUT_DIR)

        # List all JSON files in the output directory which are considered as conversation history files
        conversation_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]

        # Insert an option at the start of the list for UI purposes, possibly to serve as a 'select' prompt
        conversation_files.insert(0, "")

        # Add a section in the sidebar for displaying conversation history
        self.st.sidebar.subheader("Conversation History")
        # Allow the user to select a conversation history file from a dropdown list in the sidebar
        selected_conversation = self.st.sidebar.selectbox(
            "Select a conversation", conversation_files, index=0
        )

        # Check if a conversation file was selected (not the blank option inserted earlier)
        if selected_conversation:
            # Construct the full path to the selected conversation file
            conversation_file = os.path.join(OUTPUT_DIR, selected_conversation)
            # Open and load the conversation JSON data
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)

            # Extract the conversation ID from the selected filename for state tracking
            self.st.session_state["conversation_id"] = selected_conversation.split("_")[
                0
            ]

            # Load the conversation data into the session state for display
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = conversation_data

    def chat(self, prompt):
        """
        Handles sending a prompt to the selected language model and displaying the response in the chat interface.

        Args:
            prompt (str): The user's question or prompt for the language model.

        Returns:
            The response from the language model to the provided prompt.
        """
        # Check if there's an ongoing conversation, otherwise initialize
        if "conversation_id" in self.st.session_state:
            # Use the existing conversation ID to track chat history
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
        else:
            # If no conversation is active, generate a new ID and initialize chat history
            self.st.session_state["conversation_id"] = str(datetime.datetime.now())
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
            self.st.session_state[chat_history_key] = []

        # Iterate through the stored chat history and display it
        for message in self.st.session_state[chat_history_key]:
            role = message["role"]
            if role == "user":
                # Display user's messages with a specific format
                with self.st.chat_message("user"):
                    question = message["content"]
                    self.st.markdown(f"{question}", unsafe_allow_html=True)
            elif role == "assistant":
                # Display assistant's responses with a different format
                with self.st.chat_message("assistant"):
                    self.st.markdown(message["content"], unsafe_allow_html=True)

        # Check if there is a new prompt from the user
        if prompt:
            # Add the new prompt to the chat history
            self.st.session_state[chat_history_key].append(
                {"content": prompt, "role": "user"}
            )
            # Display the prompt in the chat UI
            with self.st.chat_message("user"):
                self.st.write(prompt)
            # Prepare the messages for the language model by collecting all messages from the chat history
            messages = [
                dict(content=message["content"], role=message["role"])
                for message in self.st.session_state[chat_history_key]
            ]
            # Fetch the response from the language model using the connector
            with self.st.chat_message("assistant"):
                chat_box = self.st.empty()  # Placeholder for the model's response
                # Use the LLM connector to stream the model's response based on the chat history
                response_message = chat_box.write_stream(
                    self.llmConnector.llm_stream(
                        self.selected_model[0], self.selected_model[1], messages
                    )
                )
            # Append the model's response to the chat history
            self.st.session_state[chat_history_key].append(
                {"content": f"{response_message}", "role": "assistant"}
            )
            # Save the conversation to a JSON file
            self.save_conversation()
            # Return the response message to be displayed in the chat UI
            return response_message

    def save_conversation(self):
        """
        Saves the current conversation to a JSON file, allowing for persistence of chat history.
        """
        conversation_id = self.st.session_state["conversation_id"]
        conversation_key = f"chat_history_{conversation_id}"
        conversation_chat = self.st.session_state[conversation_key]
        filename = f"{conversation_id}.json"

        # Check if there's any conversation to save
        if conversation_chat:
            # Prepare the file path for saving the conversation
            conversation_file = os.path.join(self.OUTPUT_DIR, filename)

            # Save the updated conversation back to the file
            with open(conversation_file, "w") as f:
                json.dump(conversation_chat, f, indent=4)
                self.st.success(f"Conversation saved to {conversation_file}")
