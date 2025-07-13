# M3Eval - MiroMind Mathematical Evaluation

This framework evaluates language models on mathematical reasoning tasks including Math-500, AIME 2024, and AIME 2025 datasets.

## Project Structure

```
m3eval/
├── main.py          # Main evaluation script with argument parsing and orchestration
├── general_utils.py # General utilities for model loading, prompt formatting, and I/O
├── eval_utils.py    # Evaluation functions for different datasets
├── math_utils.py    # Mathematical verification and answer comparison utilities
├── prompts.py       # Prompt templates for mathematical reasoning
└── data/           # Dataset storage directory
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- vLLM library for efficient inference
- HuggingFace Transformers
- pandas, datasets, torch
- others

## Usage

## Supported Datasets

### Math-500
- **Source**: HuggingFace H4/MATH-500
- **Type**: Mathematical word problems
- **Evaluation**: Match scoring with mathematical verification

### AIME 2024/2025
- **Source**: Local data (data/aime24, data/aime25)
- **Type**: American Invitational Mathematics Examination problems
- **Evaluation**: Numeric comparison with 3-digit formatting

## Key Features

### Multiple Evaluation Methods
- **Math-500**: Uses sophisticated mathematical verification with multiple fallback methods
- **AIME**: Numeric comparison with proper formatting
- **Error Analysis**: Automatic generation of error case files

### Flexible Prompting
- **Default System Message**: Simple user prompts
- **Custom Reasoning**: Chain-of-thought with `<think>` tags
- **System vs User**: Reasoning instructions in system or user messages
- **Special Tokens**: Support for special reasoning tokens

### Performance Optimization
- **Batch Processing**: Process multiple runs simultaneously
- **Memory Management**: Efficient GPU memory utilization
- **Chunked Prefill**: Optimized attention computation
- **Tensor Parallelism**: Multi-GPU support

### Comprehensive Logging
- **Timestamped Logs**: Detailed execution logs with timestamps
- **Result Files**: CSV outputs with full evaluation details
- **Error Analysis**: Separate files for incorrect predictions
- **Accuracy Tracking**: Individual and average accuracy metrics

## Output Files

The evaluation generates several output files in `results/[model_name]/[dataset_name]/`: