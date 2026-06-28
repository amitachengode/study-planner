from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError, APIError

class Agent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", system_instruction: str = None):
        self.client = genai.Client(api_key=api_key)
        # Ordered list of models to try if the primary one is overloaded
        self.model_pool = [model_name, "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-3.5-flash", "gemini-3.5-pro"]
        self.current_model_index = 0
        self.system_instruction = system_instruction
        
        self._init_chat_session()

    def _init_chat_session(self):
        """Initializes or resets a chat session with the current active model choice."""
        config = None
        if self.system_instruction:
            config = types.GenerateContentConfig(system_instruction=self.system_instruction)
            
        active_model = self.model_pool[self.current_model_index]
        self.chat_session = self.client.chats.create(model=active_model, config=config)

    def ask(self, question: str, max_tokens: int = 10000) -> str:
        config = types.GenerateContentConfig(max_output_tokens=max_tokens)
        
        # Loop through our model pool if we hit a 503 Server Error
        while self.current_model_index < len(self.model_pool):
            active_model = self.model_pool[self.current_model_index]
            try:
                # Update the underlying chat session target model dynamically
                self.chat_session._model = active_model
                
                response = self.chat_session.send_message(
                    message=question,
                    config=config
                )
                return response.text

            except ServerError as e:
                # 503 Service Unavailable / High Demand Spike
                if e.code == 503:
                    self.current_model_index += 1
                    if self.current_model_index < len(self.model_pool):
                        # Gracefully switch to the next model in the pool on the next loop iteration
                        continue
                    else:
                        return "❌ **All available Gemini models are currently experiencing high demand.** Please wait a moment and try again later."
                return f"❌ Gemini Server Error ({e.code}): {e.message}"

            except ClientError as e:
                # 429 Resource Exhausted (Quota/Rate limits)
                if e.code == 429:
                    return "⚠️ **Resource Exhausted:** You have exceeded your current API rate limits or token quota. Please check your Google AI Studio billing dashboard or wait a minute before sending another prompt."
                
                # 401 / 403 Invalid API Key / Permissions
                if e.code in [401, 403]:
                    return "🔑 **Invalid API Key:** The provided GEMINI_API_KEY is unauthorized or invalid. Please re-verify your environment configuration variables."
                
                return f"❌ Client Error ({e.code}): {e.message}"

            except APIError as e:
                return f"❌ Network API Error: {str(e)}"
            
        return "❌ Failed to generate a response from any fallback models."