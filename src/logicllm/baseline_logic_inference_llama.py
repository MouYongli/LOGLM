import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
# from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
# from symbolic_solvers.csp_solver.csp_solver import CSP_Program
# from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
from backup_answer_generation import Backup_Answer_Generator
import re
class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.logical_programs_file_name = args.logical_programs_file_name
        self.backup_strategy = args.backup_strategy
        self.dataset_name = args.dataset_name
        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                # 'ProntoQA': Pyke_Program, 
                                # 'ProofWriter': Pyke_Program,
                                # 'LogicalDeduction': CSP_Program,
                                # 'AR-LSAT': LSAT_Z3_Program}
        }
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        with open(os.path.join(self.here, 'results', 'logical_programs', f'{self.logical_programs_file_name}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples.")
        return dataset
    
    def save_results(self, outputs):       
        with open(os.path.join(self.here, 'results', 'logical_inference', f'{self.logical_programs_file_name}.json'), encoding = 'utf-8', mode='w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program, self.dataset_name)
        
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''
    
    def extract_prover9_logic_sections(self, text):
        pattern_curly_double = r'\{\{(.*?)\}\}'
        pattern_curly_single = r'\{(.*?)\}'

        match_double = re.search(pattern_curly_double, text, re.DOTALL)
        if match_double:
            extracted_text = match_double.group(1)
        else:
            match_single = re.search(pattern_curly_single, text, re.DOTALL)
            if match_single:
                extracted_text = match_single.group(1)
            else:
                pattern_backticks = r'```(.*?)```'
                match_backticks = re.search(pattern_backticks, text, re.DOTALL)
                if match_backticks:
                    extracted_text = match_backticks.group(1)
                else:
                    extracted_text = text
        return extracted_text
    
    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # execute the logic program
            row_logic_programs = self.extract_prover9_logic_sections(example['raw_logic_programs'])
            answer, flag, error_message = self.safe_execute_program(example['id'], row_logic_programs)
            if not flag == 'success':
                error_count += 1

            # create output
            output = {'id': example['id'], 
                    'context': example['context'],
                    'question': example['question'], 
                    'answer': example['answer'],
                    'flag': flag,
                    'predicted_answer': answer}
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FOLIO')
    parser.add_argument('--logical_programs_file_name', type=str, default='Logical_rules_FOLIO_dev_llama8b_logical_programs')
    parser.add_argument('--backup_strategy', type=str, default='LLM', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./results/baseline/CoT_FOLIO_dev_llama8b.json')
    parser.add_argument('--timeout', type=int, default=5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()