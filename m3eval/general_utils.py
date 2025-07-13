# Standard library imports
import os
import logging
import json
from datetime import datetime

# Third-party imports
from vllm import LLM, SamplingParams        # vLLM for efficient inference
from transformers import AutoTokenizer      # HuggingFace tokenizer
import random

logger = logging.getLogger(__name__)

def setup_logger(saved_dir):
    """Set up logging configuration with file and console outputs."""
    # Create timestamped log file
    log_file = os.path.join(saved_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),    # Log to file
            logging.StreamHandler()           # Log to console
        ]
    )
    logger.info(f"Logging setup completed. Log file: {log_file}")

def make_dirs(args):
    """Create output directories for results."""
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model/dataset specific subdirectory
    # Note: Timestamp-based subdirectories can be enabled if needed
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # args.saved_dir = os.path.join(args.output_dir, args.model_name, args.dataset_name, current_time)
    args.saved_dir = os.path.join(args.output_dir, args.model_name, args.dataset_name)
    os.makedirs(args.saved_dir, exist_ok=True)

def save_and_log_args(args):
    """Save command line arguments to JSON file for reproducibility."""
    args_dict = vars(args)                    # Convert argparse to dict
    args_file = os.path.join(args.saved_dir, 'args.json')
    
    # Save arguments to JSON file
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Log confirmation and arguments
    logger.info(f"Arguments saved to: {args_file}")
    logger.info(f"Arguments: {args_dict}")

def load_model(model_path, 
               gpu_memory_utilization, 
               tensor_parallel_size,
               max_num_seqs=None,
               enforce_eager=False,
               enable_chunked_prefill=False,
               max_model_len=None,
               disable_custom_all_reduce=False,
               disable_sliding_window=False,
               seed=42):
    """Load LLM model and tokenizer with specified configurations."""
    logger.info(f"Loading model from {model_path}")
    llm = LLM(
        model=model_path, 
        gpu_memory_utilization=gpu_memory_utilization, 
        tensor_parallel_size=tensor_parallel_size, 
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=enable_chunked_prefill,
        max_model_len=max_model_len,
        disable_custom_all_reduce=disable_custom_all_reduce,
        disable_sliding_window=disable_sliding_window,
        seed=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded successfully")
    return llm, tokenizer

def format_math_prompts(questions, tokenizer, default_sys_msg=False, reasoning_instruction=None, use_short_cot_template=False, use_short_cot_special=False, special_token=None, reasoning_in_sys_msg=False, synthetic=False):
    """Format math problem prompts with various reasoning templates."""
    logger.info("Formatting prompts for all questions")
    
    # Branch 1: Use default system message approach
    if default_sys_msg:
        if synthetic:
            # For synthetic datasets, add explicit answer format instruction
            instruct = "\n\nReturn your final response as 'Final Answer: \\boxed{<answer>}', where <answer> is the number or mathematical expression of the solution."
            messages_list = [[ 
                {"role": "user", "content": prompt + instruct}
            ] for prompt in questions]
        else:
            if use_short_cot_template:
                # Use short chain-of-thought with custom reasoning instruction
                messages_list = [[
                    {"role": "user", "content": prompt + "\n\n" + reasoning_instruction}
                ] for prompt in questions]
            else:
                # Plain user prompt without additional instructions
                messages_list = [[ 
                    {"role": "user", "content": prompt}
                ] for prompt in questions]
    
    # Branch 2: Use custom reasoning instruction approach
    else:
        from prompts import REASONING_INSTRUCTION
        # Use default reasoning instruction if none provided
        if reasoning_instruction is None:
            reasoning_instruction = REASONING_INSTRUCTION

        if reasoning_in_sys_msg:
            # Put reasoning instruction in system message
            messages_list = [[ 
                {"role": "system", "content": reasoning_instruction},
                {"role": "user", "content": prompt}
            ] for prompt in questions]
        else:
            # Append reasoning instruction to user prompt
            messages_list = [[
                {"role": "user", "content": prompt + "\n" + reasoning_instruction}
            ] for prompt in questions]
    
    # Apply tokenizer's chat template to format messages
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    
    # Optionally append special token for chain-of-thought
    if use_short_cot_special:
        formatted_prompts = [f"{prompt}{special_token}" for prompt in formatted_prompts]
    
    logger.info(f"Successfully formatted {len(formatted_prompts)} prompts")
    return formatted_prompts

def format_gpqa_prompts(data, tokenizer, ignore_gpqa_instruction=False):
    """Format GPQA prompts with randomized answer positions."""
    # ABCD option mapping
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    # GPQA prompt generation function
    def gpqa_prompt_fn(line, ignore_gpqa_instruction=False):
        gold_index = random.randint(0, 3)  # Randomly shuffle correct answer position
        choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
        choices.insert(gold_index, line["Correct Answer"])  # Insert correct answer

        if ignore_gpqa_instruction:
            query_template = (
                "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
            )
        else:
            query_template = (
                "Answer the following multiple choice question. The last line of your response should be "
                "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
                "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
            )
        query = query_template.format(A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"])
        return query, choices, gold_index, index_to_letter[gold_index]
    
    
    # Generate GPQA prompts
    gpqa_prompts = []
    gold_indices = []
    gold_letters = []
    gold_answers = []
    for line in data:
        query, choices, gold_index, gold_letter = gpqa_prompt_fn(line, ignore_gpqa_instruction=ignore_gpqa_instruction)
        gpqa_prompts.append(query)
        gold_indices.append(gold_index)
        gold_letters.append(gold_letter)
        gold_answers.append(choices[gold_index])
    
    # Build messages list for chat template
    messages_list = [
        [{"role": "user", "content": prompt}]
        for prompt in gpqa_prompts
    ]
    
    # Format for vLLM input
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    logger.info(f"Successfully formatted {len(formatted_prompts)} prompts")
    return formatted_prompts, gold_letters, gold_answers

def format_ifeval_prompts(questions, tokenizer):
    """Format prompts for IFEval instruction-following evaluation."""
    # Extract prompt text from question objects
    messages_list = [[{"role": "user", "content": q["prompt"]}] for q in questions]
    # Apply tokenizer chat template
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    logger.info(f"Successfully formatted {len(formatted_prompts)} prompts")
    return formatted_prompts

def format_livecodebench_v5_prompts(questions, tokenizer):
    """Format prompts for LiveCodeBench v5 coding evaluation."""
    # Questions are already strings, just wrap in user message
    messages_list = [[{"role": "user", "content": q}] for q in questions]
    # Apply tokenizer chat template
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    logger.info(f"Successfully formatted {len(formatted_prompts)} prompts")
    return formatted_prompts

def generate_responses(llm, formatted_prompts, temperature, top_p, max_tokens, seed):
    """Generate responses using LLM with specified sampling parameters."""
    logger.info("Starting response generation")
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    logger.info(f"Sampling parameters: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, seed={seed}")
    responses = llm.generate(formatted_prompts, sampling_params)
    logger.info(f"Successfully generated {len(responses)} responses")
    return responses

def save_csv_results(df, saved_path):
    """Save DataFrame results to CSV file with proper encoding."""
    df.to_csv(saved_path, index=False, encoding="utf-8-sig", escapechar='\\')
    logger.info(f"CSV file saved successfully to {saved_path}")

def save_accuracies(accuracies, saved_dir):
    """Save accuracy results to text files for analysis."""
    # Save individual run accuracies
    with open(os.path.join(saved_dir, "accuracies.txt"), "w") as f:
        for accuracy in accuracies:
            f.write(f"{accuracy:.4f}\n")
    logger.info(f"Accuracies {accuracies} saved successfully to {os.path.join(saved_dir, 'accuracies.txt')}")
    
    # Calculate and save average accuracy
    average_accuracy = sum(accuracies) / len(accuracies)
    with open(os.path.join(saved_dir, "average_accuracy.txt"), "w") as f:
        f.write(f"{average_accuracy:.4f}")
    logger.info(f"Average accuracy {average_accuracy:.4f} saved successfully to {os.path.join(saved_dir, 'average_accuracy.txt')}")