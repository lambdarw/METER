import asyncio
import google.generativeai as genai
from google.generativeai.types.generation_types import GenerationConfigType

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class SyncGeminiChatVision:
    def __init__(self, model, config):
        genai.configure(api_key=config["GOOGLE_API_KEY"], transport='rest')
        self.model = genai.GenerativeModel(model_name=model)
        
    def get_response(self, message, temperature=0.2, max_tokens=1024):
        response = self.model.generate_content(message, safety_settings=safety_settings)
        # print(response)
        # print(type(response.text))
        return response.text

class SyncGeminiChat:
    def __init__(self, model, config):
        genai.configure(api_key=config["GOOGLE_API_KEY"], transport='rest')
        self.model = genai.GenerativeModel(model_name=model,
                          generation_config={"response_mime_type": "application/json"})
        
    def get_response(self, message, temperature=0.2, max_tokens=1024):
        response = self.model.generate_content(message, safety_settings=safety_settings)
        print(response)
        print(type(response.text))
        return response.text
    
class AsyncGeminiChatVision:
    def __init__(self, model, config):
        genai.configure(api_key=config["GOOGLE_API_KEY"], transport='rest')
        self.model = genai.GenerativeModel(model_name=model)
    
    async def get_response(self, messages, temperature=0.2, max_tokens=1024):
        async def gemini_reply(message):
            print(message)
            response = self.model.generate_content(message, safety_settings=safety_settings)
            
            return response.text

        response_list = [gemini_reply(message) for message in messages]
        return await asyncio.gather(*response_list)
    
class AsyncGeminiChat:
    def __init__(self, model, config):
        genai.configure(api_key=config["GOOGLE_API_KEY"], transport='rest')
        self.model = genai.GenerativeModel(model_name=model,
                          generation_config={"response_mime_type": "application/json"})
    
    async def get_response(self, messages, temperature=0.2, max_tokens=1024):
        async def gemini_reply(message):
            print(message)
            response = self.model.generate_content(message, safety_settings=safety_settings)
            
            return response.text

        response_list = [gemini_reply(message) for message in messages]
        return await asyncio.gather(*response_list)