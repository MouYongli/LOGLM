import ollama

class LlamaModel:
    def __init__(self, model_name='llama3.1', model_version='8b'):
        self.model_name = model_name
        self.model_version = model_version
    
    def generate(self, prompt, temperature=0.0, top_p=1.0, seed=47):
        response = ollama.chat(
            model=self.model_name,
            message=prompt,
            temperature=temperature,
            stream=False,
        )
        return response.message.content
        
        