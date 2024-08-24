# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
import ollama
import argparse

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, '..', '..', 'data')
results_path = os.path.join(here, '..', '..', 'results')

class LogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = results_path
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                               'ProntoQA': self.prompt_prontoqa,
                               'ProofWriter': self.prompt_proofwriter,
                               'LogicalDeduction': self.prompt_logicaldeduction, 
                               'AR-LSAT': self.prompt_arlsat,
                               'Ours': self.prompt_ours}
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        here = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(here, 'prompts', "logic" ,f'{self.dataset_name}.txt')

        # prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        # if self.dataset_name == 'AR-LSAT' and self.model_name == 'gpt-4':
        #     prompt_file = f'./models/prompts/{self.dataset_name}-long.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def prompt_folio(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
    def prompt_ours(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                output = ollama.chat(
                        model=self.model_name,
                        messages=[{'role': "user", "content": full_prompt}],
                        stream=False,
                        options={'num_ctx': 4096}
                )
                # print(full_prompt)
                programs = [output]

                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
            except:
                print('Error in generating logic programs for example: ', example['id'])

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def batch_logic_program_generation(self, batch_size=1):
        # Load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        # Split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # Create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            full_prompt_str = "\n\n".join(full_prompts)  # Combine the prompts into a single string
            output = ollama.chat(
                model=self.model_name,
                messages=[{'role': "user", "content": full_prompt_str}],  # Pass as a single message
                stream=False,
                options={'num_ctx': 4096}
            )
            # Create output
            programs = output['message']['content']
            print(programs)
            example = chunk[0]
            output = {'id': example['id'],
                    'context': example['context'],
                    'question': example['question'],
                    'answer': example['answer'],
                    'options': example['options'],
                    'raw_logic_programs': programs}
            outputs.append(output)
        # Remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(outputs)
        print(f"Generated {len(outputs)} examples.")

        # Save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name[0:4]}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="FOLIO")
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--model_name', type=str, default='llama3.1:70b')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation()