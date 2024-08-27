# -*- coding: utf-8 -*-
# @Date    : 20.08.2024
# @Author  : Bozhen Zhu
# @Desc    : Baseline generator for Folio with llama

import json
import os
from tqdm import tqdm
import argparse
import ollama
import re

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, '..', '..', 'data')
results_path = os.path.join(here, 'results', 'baseline')
prompt_path = os.path.join(here, 'prompts', 'baseline')

class LLaMA_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode
    def load_prompts(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}_llama.txt')) as f:
            prompt_template = f.read()
        return prompt_template

    def load_raw_dataset(self, split):
        file_path = os.path.join(self.data_path, self.dataset_name, f'{split}.json')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        
        return raw_dataset

    def generate(self, prompt):
        # print(prompt)
        # print('--------------')
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': "user", "content": prompt}],
            stream=False,
            options={'num_ctx': 4096}
        )

        try:
            pattern = r'\{\s*(.*)\s*\}'
            match = re.search(pattern, response['message'].get('content'), re.DOTALL)
            code = match.group(1)
            code = re.sub(r'\\n\s*', '', code)
            code = '{' + code.strip() + '}'
            json_data = json.loads(code)
        except:
            pattern = r'"correct_option":\s*"[A-Z]"'
            match = re.search(pattern, response['message'].get('content'), re.DOTALL)
            code = match.group(0)
            code = re.sub(r'\\n\s*', '', code)
            code = '{' + code.strip() + '}'
            json_data = json.loads(code)
            if self.mode == "CoT":
                json_data['reasoning'] = "failed to extract reasoning"
        # finally:
        #     import pdb
        #     pdb.set_trace()
        return json_data
    
    def batch_reasoning_graph_generation(self, batch_size=10):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        outputs = []
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]

        for chunk in tqdm(dataset_chunks):
            for sample in chunk:
                try:
                    prompt_template = self.load_prompts()
                    context = sample['context'].strip()
                    question = sample['question'].strip()
                    options = '\n'.join([opt.strip() for opt in sample['options']])
                    full_prompt = prompt_template.format(context=context, question=question, options=options)
                    output = self.generate(full_prompt)
                    dict_output = self.update_answer(sample, output)
                    print(dict_output)
                    outputs.append(dict_output)
                except Exception as e:
                    print(f'Error in generating example {sample["id"]}: {e}')
        file_path = os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_llama.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def update_answer(self, sample, output):
        if self.mode == "Direct":
            dict_output = {'id': sample['id'], 
                            'question': sample['question'], 
                            'answer': sample['answer'], 
                            # 'predicted_reasoning': output['reasoning'],
                            'predicted_answer': output['correct_option'],}
        elif self.mode == "CoT":
            dict_output = {'id': sample['id'], 
                    'question': sample['question'], 
                    'answer': sample['answer'], 
                    'predicted_reasoning': output['reasoning'],
                    'predicted_answer': output['correct_option'],}         
        return dict_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--dataset_name', type=str, default="ProntoQA")
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default=results_path)
    parser.add_argument('--demonstration_path', type=str, default=prompt_path)
    parser.add_argument('--model_name', type=str, default='llama3:8b')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str, default="CoT")
    parser.add_argument('--max_new_tokens', type=int, default = 4096)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    llama_reasoning = LLaMA_Reasoning_Graph_Baseline(args)
    llama_reasoning.batch_reasoning_graph_generation(batch_size=10)
