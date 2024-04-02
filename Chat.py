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

    def __init__(self, output_dir="llm_conversations", health_check_enabled=False):
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
        self.health_check_enabled = health_check_enabled  # Enable health check
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
        # Set chat parameters
        self.params = self.chat_params()  # Chat parameters

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
        # Check the health status of the LLM providers if enabled
        if self.health_check_enabled:
            health_check = self.llmConnector.health_check()  # Check health status
            local_models_names = [
                model["name"] for model in self.LOCAL_MODELS if health_check["ollama"]
            ]  # Local models
            online_models_names = [
                model["name"]
                for model in self.ONLINE_MODELS
                if health_check[model["provider"]]
            ]  # Online models
        else:
            # Compile lists of available models
            local_models_names = [
                model["name"] for model in self.LOCAL_MODELS
            ]  # Local models
            online_models_names = [
                model["name"] for model in self.ONLINE_MODELS
            ]  # Online models

        # Combine all models
        model_names = local_models_names + online_models_names

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
            with self.st.expander("Model Details"):
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

        # Sort the conversation files by modification time in descending order
        conversation_files = sorted(
            conversation_files,
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
            reverse=True,
        )

        # Insert an option at the start of the list for UI purposes, possibly to serve as a 'select' prompt
        conversation_files.insert(0, "")

        def format_id(id):
            date = id.split(".")[0]
            return f"{date}"

        # Add a section in the sidebar for displaying conversation history
        self.st.sidebar.subheader("Conversation History")
        # Allow the user to select a conversation history file from a dropdown list in the sidebar
        selected_conversation = self.st.sidebar.selectbox(
            "Select a conversation", conversation_files, index=0, format_func=format_id
        )

        # Check if a conversation file was selected (not the blank option inserted earlier)
        if selected_conversation:
            # Construct the full path to the selected conversation file
            conversation_file = os.path.join(OUTPUT_DIR, selected_conversation)

            # Display the last modified time of the selected conversation file
            last_modified = datetime.datetime.fromtimestamp(
                os.path.getmtime(conversation_file)
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.st.sidebar.write(f"Last update: {last_modified}")

            # Open and load the conversation JSON data
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)

            # Extract the conversation ID from the selected filename for state tracking
            self.st.session_state["conversation_id"] = selected_conversation.split(".")[
                0
            ]

            # Load the conversation data into the session state for display
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = conversation_data

            self.st.session_state["chat_params"] = {
                "history_length": 5,
                "similarity_threshold": 0.5,
                "max_tokens": 2400,
                "system_prompt": "",
                "conversation_language": "None",
            }

            # Load the system prompt from the conversation history
            for message in conversation_data:
                if message["role"] == "system":
                    system_prompt = message["content"]
                    self.st.session_state["chat_params"] = {
                        "history_length": 5,
                        "similarity_threshold": 0.5,
                        "max_tokens": 2400,
                        "system_prompt": system_prompt,
                        "conversation_language": "None",
                    }

    def chat_params(self):
        """
        Displays chat parameters in the sidebar for the user to adjust the language model's behavior.

        Returns:
            A dictionary containing the chat parameters set by the user.
        """
        self.st.sidebar.subheader("Chat Parameters")

        # Load the chat parameters from the session state if available
        if "chat_params" in self.st.session_state:
            chat_params = self.st.session_state["chat_params"]
        else:
            # Set default chat parameters
            chat_params = {
                "history_length": 5,
                "similarity_threshold": 0.5,
                "max_tokens": 2400,
                "system_prompt": "",
                "conversation_language": "None",
            }

        # System prompt
        system_prompt = self.st.sidebar.text_area(
            key="system_prompt",
            label="System Prompt",
            value=chat_params["system_prompt"],
        )

        # Conversation Language
        if chat_params["conversation_language"] == "English":
            index = 0
        elif chat_params["conversation_language"] == "French":
            index = 1
        else:
            index = 2
        conversation_language = self.st.sidebar.selectbox(
            key="conversation_language",
            label="Enable Similarity Check",
            options=["English", "French", "None"],
            index=index,
        )

        # Similarity threshold
        similarity_threshold = self.st.sidebar.number_input(
            key="similarity_threshold",
            label="Similarity Threshold",
            value=chat_params["similarity_threshold"],
        )

        # History length
        history_length = self.st.sidebar.number_input(
            key="history_length",
            label="Number of Q&As to consider",
            value=chat_params["history_length"],
        )

        # Max tokens output
        max_tokens = self.st.sidebar.number_input(
            key="max_tokens", label="Max Tokens Output", value=chat_params["max_tokens"]
        )

        # return chat parameters
        return {
            "history_length": history_length,
            "similarity_threshold": similarity_threshold,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "conversation_language": conversation_language,
        }

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
                params = self.params
                # Use the LLM connector to stream the model's response based on the chat history``
                response_message = chat_box.write_stream(
                    self.llmConnector.llm_stream(
                        provider=self.selected_model[0],
                        model_name=self.selected_model[1],
                        messages_history=messages,
                        user_prompt=prompt,
                        language=params["conversation_language"],
                        history_length=params["history_length"],
                        similarity_threshold=params["similarity_threshold"],
                        max_tokens=params["max_tokens"],
                        system_prompt=params["system_prompt"],
                    )
                )

            # Modify or add a system prompt to chat history
            if params["system_prompt"]:
                # Check if a system prompt is already present in the chat history
                system_prompt_exists = any(
                    message["role"] == "system" for message in messages
                )
                # If a system prompt is not present, add it to the chat history
                if not system_prompt_exists:
                    self.st.session_state[chat_history_key].append(
                        {"content": params["system_prompt"], "role": "system"}
                    )
                # If a system prompt is present, update it in the chat history
                else:
                    for message in messages:
                        if message["role"] == "system":
                            message["content"] = params["system_prompt"]

            # Add the new prompt to the chat history
            self.st.session_state[chat_history_key].append(
                {"content": prompt, "role": "user"}
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
