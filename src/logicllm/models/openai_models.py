import os
from openai import AzureOpenAI

class OpenAIModel:
    def __init__(self, openai_endpoint=None, deployment_name="gpt-4", openai_api_key=None):
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") if openai_endpoint is None else openai_endpoint
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY") if openai_api_key is None else openai_api_key
        self.deployment_name = deployment_name
        self.client = AzureOpenAI(
            api_key=azure_openai_api_key,  
            api_version="2024-02-01",
            azure_endpoint = azure_openai_endpoint
        )

    def generate(self, prompt, temperature=0.0, top_p=1.0, stop_words="------"):
        response = self.client.generate_text(
            model=self.deployment_name,
            message=prompt,
            temperature=temperature,
            top_p=top_p,
            stop=stop_words
        )
        return response.choices[0].message.content
        
        