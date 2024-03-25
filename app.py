from Chat import Chat
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

if __name__ == "__main__":
    # Create a new chat instance and run the chat loop
    chat = Chat(output_dir="llm_conversations")
    chat.run()
