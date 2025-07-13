# Standard library imports
import os
import logging
import argparse
import random
import subprocess
import sys
import gc

# Third-party imports
import pandas as pd
import re
from datasets import load_dataset, load_from_disk
from vllm.distributed.parallel_state import destroy_model_parallel
import torch


# Local imports - utility functions for model evaluation
from general_utils import (
    setup_logger,        # Set up logging configuration
    load_model,          # Load LLM model and tokenizer
    save_and_log_args,   # Save arguments to file
    format_math_prompts, # Format prompts for math problems
    generate_responses,  # Generate model responses
    make_dirs,          # Create output directories
    save_csv_results,   # Save results to CSV
    save_accuracies,    # Save accuracy metrics
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for math evaluation script."""
    parser = argparse.ArgumentParser(description='Math 500 Evaluation Script')
    
    # Model configuration - Required model settings
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name identifier for the model')
    
    # Dataset configuration - Input data settings 
    parser.add_argument('--dataset_name', type=str, default="math500", help='Dataset name (math500, aime24, aime25)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Custom dataset path (overrides default)')
    parser.add_argument("--nrows", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument('--output_dir', type=str, default="results/", help='Directory to save results')
    
    # Inference configuration - Model generation settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory usage ratio')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallel')
    parser.add_argument('--temperature', type=float, default=0.60, help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling threshold')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Maximum output tokens per response')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of evaluation runs")
    
    # Prompt configuration - How to format prompts
    parser.add_argument("--default_sys_msg", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use default system message")
    parser.add_argument("--reasoning_instruction", type=str, default=None, help="Custom reasoning instruction")
    parser.add_argument("--use_short_cot_template", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use short chain-of-thought template")
    parser.add_argument("--use_short_cot_special", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use CoT with special token")
    parser.add_argument("--special_token", type=str, default=None, help="Special token to append")
    parser.add_argument("--reasoning_in_sys_msg", type=lambda x: (str(x).lower() == 'true'), default=False, help="Put reasoning instruction in system message")
    parser.add_argument("--synthetic", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use synthetic dataset format")
    
    # Optimization configuration - Performance settings
    parser.add_argument("--batch_multiple_runs", type=lambda x: (str(x).lower() == 'true'), default=False, help="Batch multiple runs together")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Max sequences in batch")
    parser.add_argument("--enable_chunked_prefill", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable chunked prefill optimization")
    parser.add_argument("--enforce_eager", type=lambda x: (str(x).lower() == 'true'), default=False, help="Disable CUDA graph optimization")
    parser.add_argument("--ignore_gpqa_instruction", type=lambda x: (str(x).lower() == 'true'), default=False, help="Skip GPQA-specific instructions")
    parser.add_argument("--disable_custom_all_reduce", type=lambda x: (str(x).lower() == 'true'), default=False, help="Disable custom all-reduce")
    parser.add_argument("--disable_sliding_window", type=lambda x: (str(x).lower() == 'true'), default=False, help="Disable sliding window attention")
    
    return parser.parse_args()


def load_data(args):
    """Load dataset based on the specified dataset name."""
    # Dataset path mappings for supported datasets
    DATASET_PATHS = {
        "math500": "HuggingFaceH4/MATH-500",  # Math problems from HuggingFace
        "aime24": "data/aime24",              # AIME 2024 competition problems
        "aime25": "data/aime25",              # AIME 2025 competition problems
    }
    
    # Use custom path if provided, otherwise use default mapping
    dataset_path = args.dataset_path or DATASET_PATHS.get(args.dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {args.dataset_name} not found and no path provided")
    logger.info(f"Loading dataset from {dataset_path}")
    
    def get_data_and_questions(dataset, key, nrows):
        """Extract data and questions from dataset for given key and row count."""
        data = [dataset[key][i] for i in range(nrows)]          # Full data entries
        questions = [dataset[key][i]["problem"] for i in range(nrows)]  # Problem text only
        return data, questions

    # Load different datasets with appropriate methods
    if args.dataset_name == "math500":
        # Math500 uses HuggingFace datasets format
        dataset = load_dataset(dataset_path)
        nrows = args.nrows if args.nrows is not None else len(dataset["test"])
        data, questions = get_data_and_questions(dataset, "test", nrows)

    elif args.dataset_name == "aime24":
        # AIME24 uses local disk format
        dataset = load_from_disk(dataset_path)
        nrows = args.nrows if args.nrows is not None else len(dataset["train"])
        data, questions = get_data_and_questions(dataset, "train", nrows)

    elif args.dataset_name == "aime25":
        # AIME25 uses local disk format with default config
        dataset = load_from_disk(dataset_path, "default")
        nrows = args.nrows if args.nrows is not None else len(dataset["train"])
        data, questions = get_data_and_questions(dataset, "train", nrows)

    else:
        raise ValueError(f"Dataset {args.dataset_name} not found")

    if questions:
        logger.info(f"Loaded {len(questions)} questions from dataset")
    return data, questions


def get_eval_fn(dataset_name):
    """Get appropriate evaluation function based on dataset name."""
    eval_fn_mapping = {
        "math500": "eval_math_500",
        "aime24": "eval_aime24", 
        "aime25": "eval_aime25"
    }
    
    if dataset_name not in eval_fn_mapping:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(eval_fn_mapping.keys())}")
    
    from eval_utils import eval_math_500, eval_aime24, eval_aime25
    return locals()[eval_fn_mapping[dataset_name]]


def prepare_df(data, formatted_prompts, generated_texts, prompt_token_nums=None, output_token_nums=None):
    """Prepare results DataFrame with generated texts and token counts."""
    # Create DataFrame from original dataset
    df = pd.DataFrame(data)
    
    # Add model outputs and metrics
    df["generated_text"] = generated_texts        # Model responses
    df["prompt_token_num"] = prompt_token_nums    # Input token count
    df["output_token_num"] = output_token_nums    # Output token count
    df["formatted_prompt"] = formatted_prompts   # Full formatted prompts
    
    return df


def evaluate_and_save(args, data, formatted_prompts, generated_texts, gold_answers, gold_letters, run_idx, prompt_token_nums, output_token_nums):
    """Evaluate generated texts and save results for a single run."""
    # Prepare DataFrame with all results
    df = prepare_df(data, formatted_prompts, generated_texts, prompt_token_nums, output_token_nums)
    
    # Create output path for this run
    output_base_path = os.path.join(args.saved_dir, f"run{run_idx}_inference")
    
    # Run evaluation and save results
    accuracy = run_standard_evaluation(df, output_base_path, args.dataset_name)
    
    return accuracy



def run_ifeval_evaluation(output_path, saved_dir):
    """Runs the ifeval evaluation script and extracts accuracy."""
    command = [
        sys.executable, "if_eval/evaluation_main.py",
        "--input_data=data/if_eval/input_data.jsonl",
        f"--input_response_data={output_path}",
        f"--output_dir={saved_dir}"
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    logging.info(f"STDOUT: {result.stdout}")
    logging.info(f"STDERR: {result.stderr}")
    
    return extract_strict_accuracy(result.stdout) * 100 if result.stdout else None

def extract_strict_accuracy(log_text):
    """Extracts strict accuracy score from log text."""
    pattern = r"eval_results_strict.jsonl Accuracy Scores:\n.*?prompt-level: ([\d\.]+)"
    match = re.search(pattern, log_text, re.DOTALL)
    return float(match.group(1)) if match else None

def run_standard_evaluation(df, output_base_path, dataset_name):
    """Run standard evaluation and save results with error analysis."""
    # Save raw inference results
    save_csv_results(df, f"{output_base_path}.csv")
    
    # Get appropriate evaluation function for dataset
    eval_fn = get_eval_fn(dataset_name)
    accuracy, df = eval_fn(df)
    logging.info(f"Accuracy: {accuracy:.2f}%")
    
    # Save evaluation results and error examples
    save_csv_results(df, f"{output_base_path}_eval_results.csv")           # All results with scores
    save_csv_results(df[df["correct"] == 0], f"{output_base_path}_error_examples.csv")  # Failed cases only
    
    return accuracy


def main():
    """Main execution function that orchestrates the entire evaluation process."""
    # Step 1: Initialize configuration and setup
    args = parse_args()                    # Parse command line arguments
    make_dirs(args)                        # Create output directories
    setup_logger(args.saved_dir)          # Configure logging
    save_and_log_args(args)               # Save args to file for reproducibility

    # Step 2: Check if evaluation already completed
    if os.path.exists(os.path.join(args.saved_dir, 'average_accuracy.txt')):
        logger.info("Average accuracy file already exists, skipping generation")
        return None

    # Step 3: Set random seed for reproducibility
    args.seed = random.randint(0, 10000)

    # Step 4: Initialize model and tokenizer
    llm, tokenizer = load_model(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_tokens,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        disable_sliding_window=args.disable_sliding_window,
        seed=args.seed
    )

    # Step 5: Load dataset and extract questions
    data, questions = load_data(args)
    gold_letters, gold_answers = None, None  # Initialize gold standard (for some datasets)

    # Step 6: Format prompts for the model
    formatted_prompts = format_math_prompts(
        questions=questions, 
        tokenizer=tokenizer, 
        default_sys_msg=args.default_sys_msg, 
        reasoning_instruction=args.reasoning_instruction, 
        use_short_cot_template=args.use_short_cot_template, 
        use_short_cot_special=args.use_short_cot_special, 
        special_token=args.special_token, 
        reasoning_in_sys_msg=args.reasoning_in_sys_msg, 
        synthetic=args.synthetic
    )

    # Step 7: Log first prompt for debugging
    logger.info("-"*100)
    logger.info(formatted_prompts[0])
    logger.info("-"*100)

    # Step 8: Run evaluations (batch or sequential mode)
    if args.batch_multiple_runs:
        original_length = len(formatted_prompts)
        formatted_prompts_all = formatted_prompts * args.num_runs
        logger.info(f"Running {args.num_runs} runs in batch mode with {original_length}*{args.num_runs} = {len(formatted_prompts_all)} prompts")

        responses_all = generate_responses(
            llm=llm,
            formatted_prompts=formatted_prompts_all,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=None # set to None to sample data, else num_runs will be all the same
        )
        generated_texts_all = [response.outputs[0].text for response in responses_all]
        prompt_token_nums_all = [len(response.prompt_token_ids) for response in responses_all]
        output_token_nums_all = [len(response.outputs[0].token_ids) for response in responses_all]

        destroy_model_parallel()
        del llm
        del responses_all
        gc.collect()
        torch.cuda.empty_cache()

        # Split generated texts into individual runs for evaluation
        assert len(generated_texts_all) == len(formatted_prompts_all) == args.num_runs * original_length
        accuracies = []
        for run_idx in range(args.num_runs):
            # Extract data for current run
            start_idx = run_idx * original_length
            end_idx = (run_idx + 1) * original_length
            generated_texts = generated_texts_all[start_idx:end_idx]
            formatted_prompts = formatted_prompts_all[start_idx:end_idx]
            prompt_token_nums = prompt_token_nums_all[start_idx:end_idx]
            output_token_nums = output_token_nums_all[start_idx:end_idx]
            # responses = responses_all[run_idx * original_length:(run_idx + 1) * original_length]
            accuracy = evaluate_and_save(args, data, formatted_prompts, generated_texts, gold_answers, gold_letters, run_idx, prompt_token_nums, output_token_nums)
            accuracies.append(accuracy)
            logger.info(f"Current average accuracy: {sum(accuracies) / len(accuracies):.2f}%")
        save_accuracies(accuracies, args.saved_dir)
    else:
        accuracies = []
        for run_idx in range(args.num_runs):
            responses = generate_responses(
                llm=llm,
                formatted_prompts=formatted_prompts,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + run_idx
            )

            generated_texts = [response.outputs[0].text for response in responses]
            prompt_token_nums = [len(response.prompt_token_ids) for response in responses]
            output_token_nums = [len(response.outputs[0].token_ids) for response in responses]

            destroy_model_parallel()
            del llm
            del responses
            gc.collect()
            torch.cuda.empty_cache()
            
            accuracy = evaluate_and_save(args, data, formatted_prompts, generated_texts, gold_answers, gold_letters, run_idx, prompt_token_nums, output_token_nums)
            accuracies.append(accuracy)
            logger.info(f"Current average accuracy: {sum(accuracies) / len(accuracies):.2f}%")

    save_accuracies(accuracies, args.saved_dir)


if __name__ == "__main__":
    main()
