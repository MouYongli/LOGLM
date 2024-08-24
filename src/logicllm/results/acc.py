import json

file_path = r'/DATA1/bzhu/LogLM/src/logicllm/results/logical_inference/Combined_FOLIO_dev_llama70b.json'

# Initialize counters for each flag
parsing_error_count = 0
success_count = 0
execution_error_count = 0
total_count = 0

# Read the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    
    # Iterate through each item in the JSON data
    for item in data:
        total_count += 1
        flag = item.get('flag', '').lower()
        
        if flag == 'parsing error':
            parsing_error_count += 1
        elif flag == 'success':
            success_count += 1
        elif flag == 'execution error':
            execution_error_count += 1

# Calculate the rates for each flag
parsing_error_rate = (parsing_error_count / total_count) * 100
success_rate = (success_count / total_count) * 100
execution_error_rate = (execution_error_count / total_count) * 100

# Print the results
print(f"Parsing Error Rate: {parsing_error_rate:.2f}%")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Execution Error Rate: {execution_error_rate:.2f}%")
