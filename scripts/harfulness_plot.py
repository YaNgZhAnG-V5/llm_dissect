import re
import pandas as pd
import ast

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        return s.strip("[]").split(", ")

def extract_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract input prompts and generated texts
    matches = re.findall(r'\[Input prompt\]: (.*?)\n\[Generated text\]: (.*?)\n(?=2024|$)', content, re.DOTALL)
    
    input_prompts = []
    generated_texts = []
    for match in matches:
        input_prompts.append(safe_literal_eval(match[0]))
        generated_texts.append(safe_literal_eval(match[1]))

    # Extract harmfulness scores
    harmfulness_scores = re.findall(r'Harumfulness score: (\[.*?\])', content)
    harmfulness_scores = [safe_literal_eval(score) for score in harmfulness_scores]

    # Extract layer and performance
    layer_performance = re.findall(r'layer: (.*?), performance: (.*)', content)

    # Flatten the lists
    input_prompts_flat = [item for sublist in input_prompts for item in sublist]
    generated_texts_flat = [item for sublist in generated_texts for item in sublist]
    harmfulness_scores_flat = [item for sublist in harmfulness_scores for item in sublist]

    # Find the minimum length
    min_length = min(len(input_prompts_flat), len(generated_texts_flat), len(harmfulness_scores_flat))

    # Truncate lists to the minimum length
    input_prompts_flat = input_prompts_flat[:min_length]
    generated_texts_flat = generated_texts_flat[:min_length]
    harmfulness_scores_flat = harmfulness_scores_flat[:min_length]

    # Create a dataframe
    df = pd.DataFrame({
        'Input Prompt': input_prompts_flat,
        'Generated Text': generated_texts_flat,
        'Harmfulness Score': harmfulness_scores_flat,
        'Layer': layer_performance[0][0],
        'Performance': float(layer_performance[0][1])
    })

    return df

# Path to your log file
# log_file_path = 'path_to_your_log_file.log'
log_file_path = 'workdirs/runs/240729_1420.log'

# Extract information and create dataframe
df = extract_info(log_file_path)

# Save to Excel
excel_file_path = 'output.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Excel file created: {excel_file_path}")