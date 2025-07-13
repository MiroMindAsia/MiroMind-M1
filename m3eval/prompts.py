# Prompt templates for mathematical reasoning tasks

# Default reasoning instruction for math problems
# Used for math500, aime24, aime25 datasets in user prompt (not system message)
REASONING_INSTRUCTION = "Let's think step by step and enclose the reasoning process within <think> and </think> tags. The final result in the answer MUST BE within \\boxed{}."

# Alternative system message approach (currently disabled)
# Previous default system message for reasoning, used for math500, aime24, aime25 in system message
# REASONING_PROMPT_IN_SYS_MSG = "You are designed to provide well-reasoned responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags. The final result in the answer MUST BE within \\boxed{}."