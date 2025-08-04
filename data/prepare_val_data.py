import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    dataset1 = datasets.load_dataset("Maxwell-Jia/AIME_2024", trust_remote_code=True)
    dataset2 = datasets.load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)
    dataset3 = datasets.load_dataset("math-ai/aime25", trust_remote_code=True)

    test1 = dataset1["train"]
    test2 = dataset2["test"]
    test3 = dataset3["test"]

    instruction = "Let's think step by step and enclose the reasoning process within <think> and </think> tags. The final result in the answer MUST BE within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn1(split):

        def process_fn(example, idx):
            question = example.pop('Problem')
            solution = str(example.pop('Answer')).strip()
            data = {
                "data_source": "aime24",
                "prompt": [
                    {
                        "role": "user",
                        "content": question + "\n" + instruction
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn
    
    def make_map_fn2(split):

        def process_fn(example, idx):
            question = example.pop('problem')
            solution = str(example.pop('answer')).strip()
            data = {
                "data_source": "math500",
                "prompt": [
                    {
                        "role": "user",
                        "content": question + "\n" + instruction
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn
    
    def make_map_fn3(split):
        def process_fn(example, idx):
            question = example.pop('problem')
            solution = str(example.pop('answer')).strip()
            data = {
                "data_source": "aime25",
                "prompt": [
                    {
                        "role": "user",
                        "content": question + "\n" + instruction
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        
        return process_fn

    test_dataset1 = test1.map(function=make_map_fn1('test'), with_indices=True)
    test_dataset2 = test2.map(function=make_map_fn2('test'), with_indices=True)
    test_dataset3 = test3.map(function=make_map_fn3('test'), with_indices=True)

    # Combine all test datasets into one
    test_dataset = datasets.concatenate_datasets([test_dataset1, test_dataset2, test_dataset3])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
