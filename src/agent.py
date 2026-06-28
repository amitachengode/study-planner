from google import genai
from google.genai import types

class Agent:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", system_instruction: str = None):
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        
        # Optional: Set a system instruction to give the agent a permanent persona/rules
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            
        # Initialize the persistent native chat session
        self.chat_session = self.client.chats.create(model=self.model, config=config)

    def ask(self, question: str, max_tokens: int = 10000) -> str:
        """
        Sends a message to the agent while retaining conversation memory.
        """
        # Pass turn-specific runtime variables like token ceilings
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens
        )
        
        # The SDK automatically handles appending this question and the model's response
        response = self.chat_session.send_message(
            message=question,
            config=config
        )
        return response.text

    def get_history(self):
        """
        Optional helper to inspect what the agent currently remembers.
        """
        return self.chat_session.get_history()
    
