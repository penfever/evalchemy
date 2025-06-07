import json
import csv
import statistics
import argparse
import os
from pathlib import Path

def process_results(json_file):
    # Read and parse JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract MTBench Average
    mtbench_res = results.get('MTBench', {})
    mtbench_avg = mtbench_res.get('Average', None)
    # Extract Alpaca Eval Length Controlled Winrate
    alpaca_eval_res = results.get('alpaca_eval', {})
    alpaca_winrate = alpaca_eval_res.get('length_controlled_winrate', None)
    # Extract MMLU Pro
    mmlu_pro = results.get('leaderboard_mmlu_pro', {}).get('acc,none', None)
    # Extract MixEval
    mixeval = results.get('MixEval', {}).get('gpt-4o-mini-2024-07-18', {}).get('metrics', {}).get("overall", None)
    # Extract MBPP
    mbpp_score = results.get('MBPP', {}).get('pass@1', None)
    # Extract MBPP+
    mbpp_plus_score = results.get('MBPPPlus', {}).get('pass@1', None)
    # Extract BBH Average
    bbh_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_bbh_') and isinstance(value, dict) and 'acc_norm,none' in value:
            bbh_scores.append(value['acc_norm,none'])
    bbh_avg = statistics.mean(bbh_scores) if bbh_scores else None
    # Extract GPQA Average
    gpqa_scores = []
    for key in ['leaderboard_gpqa_diamond', 'leaderboard_gpqa_extended', 'leaderboard_gpqa_main']:
        if key in results:
            gpqa_scores.append(results[key]['acc_norm,none'])
    gpqa_avg = statistics.mean(gpqa_scores) if gpqa_scores else None
    # Extract GPQA Diamond
    gpqa_diamond = results.get("GPQADiamond", {}).get("accuracy_avg", None)
    # Extract MATH Average
    math_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_math_') and isinstance(value, dict) and 'exact_match,none' in value:
            math_scores.append(value['exact_match,none'])
    math_avg = statistics.mean(math_scores) if math_scores else None
    # Extract MATH500 Average
    math500_avg = results.get('MATH500', {}).get('accuracy', None)
    # MUSR
    # Extract MUSR Average
    musr_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_musr_') and isinstance(value, dict) and 'acc_norm,none' in value:
            musr_scores.append(value['acc_norm,none'])
    musr_avg = statistics.mean(musr_scores) if musr_scores else None
    # IFEval
    # Extract IFEval average (using prompt-level strict accuracy)
    ifeval_score = results.get("leaderboard_ifeval", {}).get("prompt_level_strict_acc,none", None)
    # Extract IFEval second average (using instance-level loose accuracy)
    ifeval_second_score = results.get("leaderboard_ifeval", {}).get("inst_level_loose_acc,none", None)
    # LiveCodeBench
    livecodebench_score = results.get("LiveCodeBench", {}).get("accuracy_mean", None)
    # AIME
    aime24_score = results.get("AIME24", {}).get("accuracy_avg", None)
    aime25_score = results.get("AIME25", {}).get("accuracy_avg", None)
    # HumanEval
    hep_score = results.get("HumanEvalPlus", {}).get("python_pass@1", None)
    # CruxEval
    cruxeval_input_score = results.get("CruxEval", {}).get("input_pass@1", None)
    cruxeval_output_score = results.get("CruxEval", {}).get("output_pass@1", None)
    # Create output dictionary
    output = {
        'MTBench': mtbench_avg,
        'Alpaca Eval (LC)': alpaca_winrate,
        'BBH': bbh_avg,
        'GPQA': gpqa_avg,
        'MATH': math_avg,
        'MUSR': musr_avg,
        'IFEval (Prompt, Strict)': ifeval_score,
        'IFEval (Inst, Loose)': ifeval_second_score,
        'LiveCodeBench': livecodebench_score,
        'AIME24': aime24_score,
        'AIME25': aime25_score,
        'HumanEvalPlus': hep_score,
        'CruxEval (Input)': cruxeval_input_score,
        'CruxEval (Output)': cruxeval_output_score,
        'MATH500': math500_avg,
        'GPQADiamond': gpqa_diamond,
        'MBPP': mbpp_score,
        'MBPP+': mbpp_plus_score,
    }
    return output
    
    # Calculate average MUSR score
    musr_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_musr_') and isinstance(value, dict) and 'acc_norm,none' in value:
            musr_scores.append(value['acc_norm,none'])
    musr_avg = statistics.mean(musr_scores)
    
    output = {
        'MTBench': mtbench_avg,
        'Alpaca Eval (LC)': alpaca_winrate,
        'LB2_BBH': bbh_avg,
        'LB2_GPQA': gpqa_avg,
        'LB2_MATH': math_avg,
        'MATH500': math500_avg,
        'LB2_MUSR': musr_avg,
        'IFEval (Prompt Level, Strict)': ifeval_score,
        'IFEval (Instance Level, Loose)': ifeval_second_score,
        'MMLU Pro': mmlu_pro,
        'MixEval': mixeval,
        'MBPP': mbpp,
    }
    
    return output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process evaluation results JSON to CSV')
    parser.add_argument('--json_path', help='Path to the JSON file to process')
    args = parser.parse_args()
    
    # Convert path to Path object and resolve it
    json_path = Path(args.json_path).resolve()
    
    # Ensure the JSON file exists
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return
    
    try:
        # Process the results
        results = process_results(json_path)
        
        # Create output path with same name but .csv extension
        csv_path = json_path.with_suffix('.csv')
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for metric, value in results.items():
                if value is not None:
                    writer.writerow([metric, round(value, 4)])
                else:
                    continue
        
        print(f"\nResults have been saved to: {csv_path}")
        print("\nSummary of results:")
        for metric, value in results.items():
            if value is not None:
                print(f"{metric}: {round(value, 4)}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()