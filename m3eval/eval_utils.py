# Standard library imports
from typing import Callable, Literal, Sequence
import pandas as pd
import re

def eval_math_500(df):
    """Evaluate Math-500 dataset results using match scoring."""
    df = df.copy()  # Create a copy to avoid modifying the original
    
    # Ensure answers are properly formatted with boxed notation
    df["answer"] = df["answer"].apply(lambda x: f"\\boxed{{{x}}}" if not str(x).startswith("\\boxed{") else str(x))

    # Import math scoring utility
    from math_utils import compute_score

    # Compute match scores between generated text and ground truth
    df["match_score"] = df.apply(lambda row: compute_score(row["generated_text"], row["answer"]), axis=1)
    
    # Calculate overall metrics
    score = df["match_score"].sum()           # Total correct answers
    df["correct"] = df["match_score"].astype(int)  # Binary correct column
    total = len(df)                          # Total questions
    accuracy = score / total * 100 if total > 0 else 0  # Percentage accuracy
    
    return accuracy, df

def extract_boxed_answer(response):
    """Extract the last boxed answer from model response."""
    matches = re.findall(r'\\boxed{([^}]*)}', response)
    return matches[-1] if matches else "N/A"

def format_extracted_answer(answer):
    """Format extracted answer with leading zeros for AIME format (3 digits)."""
    if answer.isdigit():
        num_len = len(answer)
        if num_len == 2 and not answer.startswith("0"):
            return "0" + answer  # Pad 2-digit numbers
        elif num_len == 1:
            return "00" + answer  # Pad 1-digit numbers
    return answer

def eval_aime24(df):
    """Evaluate AIME 2024 dataset results with numeric comparison."""
    df = df.copy()
    
    # Extract answers from generated text using regex
    df["extracted_answer"] = df["generated_text"].apply(extract_boxed_answer)
    
    # Format extracted answers to 3-digit format (AIME standard)
    df["extracted_answer"] = df["extracted_answer"].apply(format_extracted_answer)
    
    # Compare extracted answers with ground truth numerically
    df["correct"] = (
        pd.to_numeric(df["extracted_answer"], errors='coerce')
        == pd.to_numeric(df["answer"], errors='coerce')
    ).astype(int)

    # Calculate accuracy percentage
    accuracy = (df["correct"].sum() / len(df) * 100)
    return accuracy, df

def eval_aime25(df):
    """Evaluate AIME 2025 dataset results with numeric comparison."""
    df = df.copy()
    
    # Extract answers from generated text using regex
    df["extracted_answer"] = df["generated_text"].apply(extract_boxed_answer)
    
    # Format extracted answers to 3-digit format (AIME standard)
    df["extracted_answer"] = df["extracted_answer"].apply(format_extracted_answer)

    # Also format ground truth answers to ensure consistent comparison
    df["answer"] = df["answer"].astype(str)
    df["answer"] = df["answer"].apply(format_extracted_answer)

    # Compare extracted answers with ground truth numerically
    df["correct"] = (
        pd.to_numeric(df["extracted_answer"], errors='coerce')
        == pd.to_numeric(df["answer"], errors='coerce')
    ).astype(int)

    # Calculate accuracy percentage
    accuracy = (df["correct"].sum() / len(df) * 100)
    return accuracy, df
