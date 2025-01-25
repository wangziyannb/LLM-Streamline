import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def load_json(file_path):
    """Helper function to load JSON files."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_std(model_result, labels):
    """Calculate standard deviation of PPL scores."""
    data = load_json(model_result)
    return [np.std([data[str(i)][f'label: {j}']['PPL'] for j in labels]) 
            for i in range(len(data))]

def get_prediction_matches(pruned_data, original_data, weights):
    """Calculate weighted sum of matching predictions."""
    matches = sum(1 for i in range(len(pruned_data)) 
                 if (pruned_data[str(i)]['prediction'] == pruned_data[str(i)]['gold'] and 
                     original_data[str(i)]['prediction'] == original_data[str(i)]['gold']) or
                    (pruned_data[str(i)]['prediction'] != pruned_data[str(i)]['gold'] and 
                     original_data[str(i)]['prediction'] != original_data[str(i)]['gold']))
    
    return sum(weights[i] for i in range(matches))

def calculate_stability(pruned_model_result, model_result, labels):
    """Calculate stability metric."""
    std_list = calculate_std(model_result, labels)
    normalized_weights = np.exp(std_list)
    normalized_weights /= normalized_weights.sum()
    
    pruned_data = load_json(pruned_model_result)
    original_data = load_json(model_result)
    
    return get_prediction_matches(pruned_data, original_data, normalized_weights)

def process_dataset_group(model_result_dir, pruned_model_dir, dataset_configs):
    """Process a group of datasets and return stability scores."""
    results = {}
    for dataset, labels in dataset_configs.items():
        model_path = Path(model_result_dir) / dataset
        pruned_path = Path(pruned_model_dir) / dataset
        
        if model_path.exists() and pruned_path.exists():
            stability = calculate_stability(str(pruned_path), str(model_path), labels)
            results[dataset] = stability
    
    return results
    
def process_cmmlu_mmlu(model_result_dir, pruned_model_dir):
    """Process CMMLU and MMLU datasets separately."""
    model_dir = Path(model_result_dir)
    pruned_dir = Path(pruned_model_dir)
    
    # Separate CMMLU and MMLU files
    cmmlu_files = sorted(model_dir.glob("cmmlu-*.json"))
    mmlu_files = sorted(model_dir.glob("lukaemon_mmlu_*.json"))
    
    # Process CMMLU
    cmmlu_total = 0
    print("\nProcessing CMMLU datasets:")
    for file_path in tqdm(cmmlu_files):
        score = calculate_stability(
            str(pruned_dir / file_path.name),
            str(file_path),
            ['A', 'B', 'C', 'D']
        )
        cmmlu_total += score
    
    cmmlu_avg = cmmlu_total / len(cmmlu_files) if cmmlu_files else 0
    print(f"CMMLU Average: {cmmlu_avg:.4f}")
    
    # Process MMLU
    mmlu_total = 0
    print("\nProcessing MMLU datasets:")
    for file_path in tqdm(mmlu_files):
        score = calculate_stability(
            str(pruned_dir / file_path.name),
            str(file_path),
            ['A', 'B', 'C', 'D']
        )
        mmlu_total += score
    
    mmlu_avg = mmlu_total / len(mmlu_files) if mmlu_files else 0
    print(f"MMLU Average: {mmlu_avg:.4f}")
    
    return {
        'cmmlu_avg': cmmlu_avg,
        'mmlu_avg': mmlu_avg
    }

def get_stability(model_result_dir, pruned_model_dir):
    """Main function to calculate stability metrics for all datasets."""
    # Configuration for different dataset groups
    basic_datasets = {
        'piqa.json': ['0', '1'],
        'BoolQ.json': ['0', '1'],
        'C3.json': ['0', '1', '2', '3'],
        'commonsense_qa.json': ['A', 'B', 'C', 'D', 'E'],
        'hellaswag.json': ['0', '1', '2', '3'],
        'race-middle.json': ['A', 'B', 'C', 'D'],
        'cmnli.json': ['contradiction', 'entailment', 'neutral'],
        'WSC.json': ['A', 'B'],
        'chid-test.json': ['0', '1', '2', '3', '4', '5', '6'],
        'race-high.json': ['A', 'B', 'C', 'D']
    }

    # Process basic datasets
    print("\nProcessing basic datasets:")
    basic_results = process_dataset_group(model_result_dir, pruned_model_dir, basic_datasets)
    for dataset, score in basic_results.items():
        print(f"{dataset}: {score:.4f}")

    # Process CMMLU and MMLU datasets
    print("\nProcessing CMMLU and MMLU datasets:")
    cmmlu_mmlu_results = process_cmmlu_mmlu(model_result_dir, pruned_model_dir)

import sys

def main(arg1, arg2):
    get_stability(arg1, arg2)
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])