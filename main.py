import ollama
import yaml
import json
import os
from huggingface_hub import login
from huggingface_hub import whoami
from datasets import load_dataset
from datetime import datetime
import glob

# Global Vars
debug = False
calc_stats = True
log_messages = []
timestamp_started = datetime.now()
timestamp_finished = datetime.now()
results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

# Function to log messages
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    log_messages.append(formatted)
    print(formatted)

# Function to connect to Hugging face, HF_TOKEN needs to be set in environment variables
def connect_hf():
    # hf_hub login
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        login(hf_token)
        user = whoami(token=hf_token)
        log(f"Connected to Hugging Face and logged in as: {user['name']}")
    except Exception as e:
        log(f"Error connecting to Hugging Face: {e}")
        exit()

# Sentiment Analysis
# Load config from config_SA.yaml
# Load configuration values from yaml file, including dataset name and split, context, prompt, llms to use and their parameters, and evaluation metrics
# For each llm runs the sentiment analysis using the prompt defined in the configuration
# It creates a log file per llm with date, run and results

def sentiment_analysis(config_file="config_SA.yaml"):
    
    # Load configuration values from yaml file
    log(f"Sentiment analysis started with config file: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log(f"Error loading config file: {e}")
        return
    dataset_name = config.get("dataset", {}).get("name")
    dataset_split = config.get("dataset", {}).get("split")
    context = config.get("context", "")
    prompt = config.get("prompt", [context])[0]  # Use the first prompt from config, or context if no prompts defined
    llms = config.get("llms", [])
    log(f"Configuration loaded: dataset={dataset_name}, split={dataset_split}, context={context}, llms={llms}")

    # Load dataset
    try:
        ds = load_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        log(f"Error loading dataset: {e}")
    log(f"Dataset {dataset_name} with split {dataset_split} loaded successfully!")
    if debug:  
        log(f"First 5 samples: {ds[:5]}")
    
    for llm in llms:
        llm_responses = []
        temperature = llm.get("temperature", "default")
        for sample in ds:
            log(f"Processing LLM: {llm['model']} with temperature {llm.get('temperature', 'default')}, prompt={prompt.replace('{text}', sample['sentence'])}")
            response = ollama.generate(
                model=llm['model'],
                prompt=prompt.replace("{text}", sample['sentence']),
                system=context[0],
                options={
                    "temperature": temperature
            }
            )
            print(response['response'])
            if debug:
                log(f"Prompt: {prompt}. Raw response: {response}")
            #check if response is 1 or 0, if not, stop and log an error
            model_response = response['response'].strip()
            if model_response not in ['0', '1']:
                log(f"Error: Unexpected response '{response['response']}' for sample {sample['idx']}")
                continue
            llm_responses.append({
                "id": sample['idx'],
                "text": sample['sentence'],
                "response": model_response,
                "golden_standard": sample['label']
            })
        # Save responses to a json file (only the fields we need)
        # create folder with experiment name and date inside results folder, and save the json file there
        experiment_folder = f"{results_folder}/{timestamp_started.strftime('%Y%m%d_%H%M%S')}_{config['experiment_name']}"
        os.makedirs(experiment_folder, exist_ok=True)
        #add date to the output file name, like date_experiment_modelname.json
        output_file = f"{experiment_folder}/{timestamp_started.strftime('%Y%m%d')}_{config['experiment_name']}_{llm['model'].replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_responses, f, indent=4, ensure_ascii=False)
        timestamp_finished = datetime.now()
        log(f"Experiment '{config['experiment_name']}' with model '{llm['model']}' completed at {timestamp_finished.strftime('%Y-%m-%d %H:%M:%S')}.")
        log(f"Time taken for model '{llm['model']}': {(timestamp_finished - timestamp_started).total_seconds()} seconds.")
        log(f"Responses saved to {output_file}")

    #End of processing, log the duration
    timestamp_finished = datetime.now()
    timestamp = timestamp_finished.strftime("%Y-%m-%d %H:%M:%S")
    duration_seconds = (timestamp_finished - timestamp_started).total_seconds()
    minutes, seconds = divmod(duration_seconds, 60)
    log(f"Processing finished at {timestamp} it took {minutes} minutes and {seconds} seconds.")
    #save log messages to a file inside the folder created previously in results, with the name of the experiment and the date, like date_experiment.log
    
    log_file = f"{experiment_folder}/{timestamp_started.strftime('%Y%m%d')}_{config['experiment_name']}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_messages))
    #If gen_stats is True, calculate statistics
    if calc_stats:
        log("Calculating statistics...")
        calculate_statistics(experiment_folder)
        log("Calculating failed response counts...")
        get_failed_count(experiment_folder)

# Function to calculate statistics, f1score, accuracy, precision, recall, time taken.
# From the json file with the responses, calculate the statistics and save them in a new json file with the same name but with _stats.json at the end, like date_experiment_modelname_stats.json
# Iterate all the json files in the results folder, and calculate the statistics for each one of them, and save them in a new json file with the same name but with _stats.json at the end, like date_experiment_modelname_stats.json
# The failed responses are logged to a separate file with the name date_experiment_modelname_failed.json, containing the id, text, response, golden_standard and error message for each failed response.
def calculate_statistics(response_dir):
    for responses_file in glob.glob(f"{response_dir}/*run*.json"):
        if responses_file.endswith('_stats.json'):
            continue
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        if responses_file.endswith('_failed.json'):
            continue
        if responses_file.endswith('.log'):
            continue
        failed_responses = []
        for r in responses:
            if str(r['response']) != str(r['golden_standard']):
                failed_responses.append({
                    "id": r['id'],
                    "text": r['text'],
                    "response": r['response'],
                    "golden_standard": r['golden_standard'],
                    "error_message": "Response does not match golden standard"
                })

        if failed_responses:
            failed_output_file = responses_file.replace('.json', '_failed.json')
            with open(failed_output_file, 'w', encoding='utf-8') as f:
                json.dump(failed_responses, f, indent=4, ensure_ascii=False)
            log(f"Failed responses saved to {failed_output_file}")

        # Calculate statistics and log failed responses
        true_positives = sum(1 for r in responses if r['response'] == '1' and r['golden_standard'] == 1)
        true_negatives = sum(1 for r in responses if r['response'] == '0' and r['golden_standard'] == 0)
        false_positives = sum(1 for r in responses if r['response'] == '1' and r['golden_standard'] == 0)
        false_negatives = sum(1 for r in responses if r['response'] == '0' and r['golden_standard'] == 1)

        accuracy = (true_positives + true_negatives) / len(responses) if responses else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        model = os.path.basename(responses_file).split('_')[2]  # Extract model name from file name
        stats = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_samples": len(responses),
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

        # Save statistics to a new json file
        stats_file = responses_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        log(f"Statistics calculated and saved to {stats_file}")

# Function to get the index of failed responses that are repeated in all models, to identify which samples are more diffucult to identify
# Keep id and strings and count of failures for each sample, and sort them by the number of failures, to identify which samples are more difficult to identify
# Include a summary of the failed samples: xx samples failed at least once, xx samples failed more than 3 times, etc.
def get_failed_count(response_dir):
    failed_counts = {}
    for failed_file in glob.glob(f"{response_dir}/*_failed.json"):
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_responses = json.load(f)
        for r in failed_responses:
            idx = r['id']
            if idx not in failed_counts:
                failed_counts[idx] = {
                    "count": 0,
                    "text": r['text'],
                    "golden_standard": r['golden_standard']
                }
            failed_counts[idx]["count"] += 1
    # Sort the failed counts by the number of failures
    sorted_failed_counts = dict(sorted(failed_counts.items(), key=lambda item: item[1]["count"], reverse=True))
    # Log summary of failed samples to the file
    total_failed_samples = len(failed_counts)
    failed_once = sum(1 for r in failed_counts.values() if r["count"] == 1)
    failed_twice = sum(1 for r in failed_counts.values() if r["count"] == 2)
    failed_more_than_2 = sum(1 for r in failed_counts.values() if r["count"] > 2)
    
    output_data = {
        "summary": {
            "total_failed_samples": total_failed_samples,
            "failed_once": failed_once,
            "failed_twice": failed_twice,
            "failed_three_or_more": failed_more_than_2
        },
        "failed_samples": sorted_failed_counts
    }

    #concat failed_summary and sorted_failed_counts and save to a json file
    failed_count_file = f"{response_dir}/failed_matchs.json"
    with open(failed_count_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    log(f"Failed counts saved to {failed_count_file}")

#Function to detect feelings, to be implemented later
#def feelings_analysis(): 

# main.py
# parameters to launch sentiment analysis or feelings analysis
# --analysis sentiment (To assess sentiment)
# --analysis feelings (To detect feelings)
# --config config_SA.yaml (To specify the configuration file, default is config_SA.yaml)
# --generate-stats <directory> (To generate statistics given a directory)
def main():
    log(f"Script started at {timestamp_started.strftime('%Y-%m-%d %H:%M:%S')}")
    connect_hf()
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--analysis":
        if len(os.sys.argv) > 2:
            analysis_type = os.sys.argv[2]
            if analysis_type == "sentiment":
                sentiment_analysis(os.sys.argv[3] if len(os.sys.argv) > 3 else "config_SA.yaml")
            elif analysis_type == "feelings":
                print("Feelings analysis not implemented yet.")
            else:
                print("Unknown analysis type. Use 'sentiment' or 'feelings'.")
        else:
            print("Please specify the analysis type after --analysis.")
    elif len(os.sys.argv) > 1 and os.sys.argv[1] == "--calculate-stats":
        if len(os.sys.argv) > 2:
            response_dir = os.sys.argv[2]
            calculate_statistics(response_dir)
        else:
            print("Please specify the directory containing the response json files after --calculate-stats.")
    elif len(os.sys.argv) > 1 and os.sys.argv[1] == "--get-failed-count":
        if len(os.sys.argv) > 2:
            response_dir = os.sys.argv[2]
            get_failed_count(response_dir)
        else:
            print("Please specify the directory containing the response json files after --get-failed-count.")
    else:
        print("No analysis type specified. Use --analysis followed by 'sentiment' or 'feelings', or use --calculate-stats followed by the directory containing response json files.")

if __name__ == "__main__":
    main()