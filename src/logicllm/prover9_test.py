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
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from symbolic_solvers.fol_solver.fol_prover9_parser import Prover9_FOL_Formula
from symbolic_solvers.fol_solver.Formula import FOL_Formula
import re
class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.logical_programs_file_name = args.logical_programs_file_name
        self.backup_strategy = args.backup_strategy
        self.dataset_name = args.dataset_name
        self.dataset = self.load_logic_programs()
        self.logic_programs = []
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                # 'ProntoQA': Pyke_Program, 
                                # 'ProofWriter': Pyke_Program,
                                # 'LogicalDeduction': CSP_Program,
                                # 'AR-LSAT': LSAT_Z3_Program}
        }
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)
    
    def load_logic_programs(self):
        error_ids = []

        with open(os.path.join(self.here, 'results', 'logical_inference', 'Logical_rules_FOLIO_dev_llama70b.json')) as f:
            data = json.load(f)
            for item in data:
                if item.get('flag') == 'parsing error':
                    error_ids.append(item.get('id'))
        print(f"Loaded {len(error_ids)} examples.")
        raw_logic_programs = []
        with open(os.path.join(self.here, 'results', 'logical_programs', 'Logical_rules_FOLIO_dev_llama70b_logical_programs.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            if item.get('id') in error_ids:
                raw_logic_programs.append(item)
        return raw_logic_programs
    

    def parse_logic_program(self):
        try:        
            # Split the string into premises and conclusion
            predicates_pattern = r'Predicates:(.*?)(?:Premises:|Query:|$)'
            premises_pattern = r'Premises:(.*?)(?:Query:|$)'
            query_pattern = r'Query:(.*)'

            predicates_match = re.search(predicates_pattern, self.logic_programs, re.DOTALL)
            premises_match = re.search(premises_pattern, self.logic_programs, re.DOTALL)
            query_match = re.search(query_pattern, self.logic_programs, re.DOTALL)

            predicates = predicates_match.group(1).strip() if predicates_match else ''
            premises_string = premises_match.group(1).strip() if premises_match else ''
            query_string = query_match.group(1).strip() if query_match else ''

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split('\n')
            query = query_string.strip().split('\n')

            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = query[0].split(':::')[0].strip()
            # print(self.logic_premises)
            # print(self.logic_conclusion)
            # print('---------------------------')
            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                # print(fol_rule)
                if fol_rule.is_valid == False:
                    print(premises)
                    return False
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            # print(fol_conclusion)

            if fol_conclusion.is_valid == False:
                return False
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True
        except Exception as e:
            print(e)
            return False
        
    def safe_execute_program(self, id, logic_program):
        program = self.parse_logic_program()
        
        # cannot parse the program
        # if program.flag == False:
        #     answer = self.backup_generator.get_backup_answer(id)
        #     return answer, 'parsing error', ''
        # # execuate the program
        # answer, error_message = program.execute_program()
        # # not executable
        # if answer is None:
        #     answer = self.backup_generator.get_backup_answer(id)
        #     return answer, 'execution error', error_message
        # # successfully executed
        # answer = program.answer_mapping(answer)
        # return answer, 'success', ''
    
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
            raw_logic_programs = self.extract_prover9_logic_sections(example['raw_logic_programs'])
            self.logic_programs = raw_logic_programs
            self.safe_execute_program(example['id'], raw_logic_programs)
            # if not flag == 'success':
            #     error_count += 1

            # # create output
            # output = {'id': example['id'], 
            #         'context': example['context'],
            #         'question': example['question'], 
            #         'answer': example['answer'],
            #         'flag': flag,
            #         'predicted_answer': answer}
            # outputs.append(output)
        
        # print(f"Error count: {error_count}")
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FOLIO')
    parser.add_argument('--logical_programs_file_name', type=str, default='Logical_rules_FOLIO_dev_llama70b_logical_programs')
    parser.add_argument('--backup_strategy', type=str, default='LLM', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./results/baseline/CoT_FOLIO_dev_llama70b.json')
    parser.add_argument('--timeout', type=int, default=5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()