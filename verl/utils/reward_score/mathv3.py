# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

from sympy import Float, Rational, Integer, Symbol
from math_verify import verify, parse
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import re


SYS_PROMPT = "Let's think step by step and enclose the reasoning process within <think> and </think> tags. The final result in the answer MUST BE within \\boxed{}."

def compute_score(solution_str, ground_truth) -> float:

    ## Discard the thinking process: only taking response after </think>
    solution_str = solution_str.replace(SYS_PROMPT, '')    
    answer_index = solution_str.rfind("</think>")
    if answer_index == -1:
        # return 0. ## no answer in response
        return {
            "score": 0.,
            "acc": False,
            "pred": "[INVALID]",
        }
    
    solution_str = solution_str[answer_index:]

    # Case 0: if response has no box, return 0 
    if solution_str.rfind("\\boxed") == -1:
        # return 0.
        return {
            "score": 0.,
            "acc": False,
            "pred": "[INVALID]",
        }
    
    # Case 1: from verl
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)

            # Remove everything before = (including = and any space after it)
            if "=" in answer:
                answer = re.sub(r'^.*=\s*', '', answer)
            if "=" in ground_truth:
                ground_truth = re.sub(r'^.*=\s*', '', ground_truth)
            
            if is_equiv(answer, ground_truth):
                # return 1.
                return {
                    "score": 1.,
                    "acc": True,
                    "pred": answer,
                }
    
    except Exception as e:
        print("ERROR from VERL:", e)

    # Case 2: apply math_verify (v0.7.0)
    gold = parse("\\boxed{" + ground_truth + "}", extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
    response = parse(solution_str, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])

    if response is not None and response != []: ## check if response is not empty else return 0
        # math verify bad case: rounding off error to 2, 3 decimal places
        try:
            if verify_with_precision(gold, response):
                # return 1.
                return {
                    "score": 1.,
                    "acc": True,
                    "pred": response[-1],
                }
            
        except Exception as e:
            print("ERROR from math_verify of all cases with naive float_rounding:", e)

        # \(\)
        # math verify bad case: "\(\\boxed{1}.\n\n \\boxed{2}\)"
        try:
            last_paragraph_without_box = re.sub(r'\\\(\s*(\\boxed\{.*?\})\s*\\\)', r"\1", solution_str)
            last_paragraph_without_box = parse(last_paragraph_without_box, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
            if verify_with_precision(gold, last_paragraph_without_box):
                # return 1.
                return {
                    "score": 1.,
                    "acc": True,
                    "pred": last_paragraph_without_box[-1],
                }
            
        except Exception as e:
            print("ERROR from math_verify(``\(\)``):", e)

        # math verify bad case: units, "%" , degree and pi from last_paragraph 
        try: ## normalize units
            last_paragraph_normalize = normalize_final_answer(solution_str)
            last_paragraph_normalize = parse(last_paragraph_normalize, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
            gold_normalize = parse("\\boxed{" + normalize_final_answer(ground_truth) + "}")
        
            if verify_with_precision(gold_normalize, last_paragraph_normalize):
                # return 1.
                return {
                    "score": 1.,
                    "acc": True,
                    "pred": last_paragraph_normalize[-1],
                }
               
        except Exception as e:
            print("ERROR from math_verify(units, \"%\", degree and pi):", e)
    
        # math verify bad case: response with roman numeral
        try: ## boolean check if response contains roman numerals
            if contains_roman_numerals(response):
                # verify response with gold answer
                if verify_roman_response(response, ground_truth):
                    # return 1.
                    return {
                        "score": 1.,
                        "acc": True,
                        "pred": response[-1],
                    }
        
        except Exception as e:
            print("ERROR from math_verify(roman numeral):", e)
        

        # # math verify bad case: response with fraction and gold answer is a float
        # try:
        #     if "%" in solution_str or "%" in ground_truth:
        #         if compare_percentage(gold, response):
        #             return 1.
        # except Exception as e:
        #     print("ERROR from math_verfy(percentage):", e)
        

        # # check for complex numbers (expressed either with a fraction or just numbers)
        # try:
        #     if verify_complex_in_fraction(gold, response):
        #         return 1.
        # except Exception as e:
        #     print("ERROR from complex number verification:", e)

        
        # # math verify bad case: response with fraction (need to normalize to float and compare)
        # try:
        #     if verify_numeric_against_fraction(gold, response, "%" in solution_str or "%" in ground_truth):
        #         return 1.
        # except Exception as e:
        #     print("ERROR from math_verfy(fraction):", e)

    # return 0.
    return {
        "score": 0.,
        "acc": False,
        "pred": "[INVALID]",
    }


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


## helper function to normalize string
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    (".}", "}"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    ("^{\\circ}", "*(3.1415926/180)"),
    ("\\pi", "*(3.1415926)"),
    (r"^\circ", "*(3.1415926/180)"),
    (r"^{\circ}", "*(3.1415926/180)"),
    ("\\%", "*(1/100)"),
    ("\\%,", "*(1/100)"),
    ("meters", ""),
]

REMOVED_EXPRESSIONS = [
    "€",
    "£",
    "₹",
    "square",
    "ways",
    "integers",
    "dollars",
    "yuan",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm^2",
    "m^2",
    "km^2",
    "cm",
    "gm",
    "mm",
    "km",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{ square units}",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "\\text{ yuan}",
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    if re.match(r'^\d*\.\d+,$', final_answer):
        final_answer = final_answer.replace(',', '')

    return final_answer


## helper function for identifying precision of answer
def find_first_small_numeric(lst, precision=0.001):
    """
    Return the first element from the list that is a numeric type (e.g. int, float,
    sympy.Integer, or sympy.Float) and for which the comparison (element < 0.001)
    evaluates to True.
    
    Parameters:
        lst (list): A list of elements.
        precision (float): The precision to use for the comparison.
    Returns:
        The first element that meets the numeric type and value condition, or None if no such element exists.
    """
    # List of acceptable numeric types.
    numeric_types = (Float, Rational, Integer)
    
    for element in lst:
        # Only consider elements that are instances of one of the numeric types.
        if isinstance(element, numeric_types):
            try:
                if element < precision:
                    return element
            except Exception:
                # If comparison fails, skip this element.
                continue
    return

def verify_with_precision(gold, response):

    # case: gold is a float < 0.001
    if find_first_small_numeric(gold, precision=0.001): 
        if verify(gold, response):
            return 1.
        
    # case: gold is [0.001, 1)
    elif find_first_small_numeric(gold, precision=1):
        if verify(gold, response, float_rounding=3):
            return 1.
        
    # case: gold is >= 1    
    if verify(gold, response, float_rounding=2):
        return 1.
    
    return 0.
    




## collections of helper functions to process roman numerals for response with gold answer.
def clean_roman_text(roman_text: str) -> str:
    """
    Clean the input text by converting to uppercase and removing any characters 
    that are not valid Roman numeral letters.
    """
    valid_symbols = set("MDCLXVI")
    return "".join(ch for ch in roman_text.upper() if ch in valid_symbols)


def convert_to_integer(roman_text: str) -> int:
    """
    Convert a Roman numeral string to an integer.
    
    This function cleans the input using clean_roman_text, then converts 
    the resulting Roman numeral to its integer value using the standard subtraction method.
    
    Parameters:
        roman_text (str): The input Roman numeral string (which may include extra characters).
    
    Returns:
        int: The integer value corresponding to the cleaned Roman numeral.
             Returns 0 if the cleaned numeral is empty.
    """
    # Clean the input text to extract a proper Roman numeral.
    cleaned = clean_roman_text(roman_text)
    if not cleaned:
        return 0

    # Mapping of Roman numerals to their integer values.
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    
    total = 0
    prev_value = 0
    
    # Process the numeral from right to left.
    for ch in reversed(cleaned):
        value = roman_dict[ch]
        # If the current value is less than the previous value, subtract it.
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total


def convert_roman_text(text: str) -> str:
    """
    Replace all valid Roman numeral tokens in the input text with their corresponding integer values.
    
    This function searches for tokens that consist solely of the Roman numeral characters
    and replaces them with the integer conversion. For example:
        "I and III" becomes "1 and 3"
        "I, II" becomes "1, 2"
    
    Parameters:
        text (str): The input string potentially containing Roman numeral tokens.
    
    Returns:
        str: The modified text with valid Roman numeral tokens replaced by their integer representations.
    """
    # This helper function is used by re.sub to replace each match.
    def replace_match(match: re.Match) -> str:
        numeral = match.group(0)
        # Ensure consistent processing by using uppercase.
        numeral = numeral.upper()
        # Convert the Roman numeral to an integer.
        integer_value = convert_to_integer(numeral)
        return str(integer_value)

    # The pattern matches whole words that consist solely of valid Roman numeral characters.
    pattern = r'\b[MDCLXVI]+\b'
    return re.sub(pattern, replace_match, text)


def verify_roman_response(response, ground_truth):
    """
    Parse the last element of the provided list from math verify parse with Roman numeral and convert it to an integer.
    
    Parameters:
        response (list): A list of items where the last element is expected to be a Roman numeral string.
                            This list is typically the result of a previous math verification parsing step.
    
    Returns:
        int: The integer value converted from the Roman numeral text.
    
    Note:
        This function relies on two helper functions:
          - convert_roman_text: which cleans or formats the Roman numeral string.
          - convert_to_integer: which performs the final conversion from the formatted Roman numeral to an integer.
    """
    gold = parse("\\boxed{" + convert_written_numbers(ground_truth) + "}")
    converted_response = convert_roman_text(response[-1])
    
    ## attempt with text box on response
    if verify(gold, parse('\\boxed{' + '\\text{' + converted_response + '}' + '}')):
        return True
    
    ## attempt without text box on response
    elif verify(gold, parse('\\boxed{' + converted_response + '}')):
        return True
    
    else:   
        return False


## check if string contains roman numerals
def contains_roman_numerals(response: str) -> bool:
    """
    Check if the given string contains a valid Roman numeral while avoiding false positives
    in mathematical expressions.

    The function first looks for indicators of math expressions (e.g., '=', arithmetic operators,
    or LaTeX-specific characters). If such characters are detected, it assumes the input is a math
    expression and returns False. Otherwise, it extracts tokens composed solely of Roman numeral
    characters and validates them against a strict Roman numeral pattern.

    Parameters:
        response (str): The input string to check.
        
    Returns:
        bool: True if a valid Roman numeral is found in normal text,
              False otherwise.
    """
    # Heuristic: if the text appears to be a math expression, skip roman numeral checks.
    # This includes common math symbols: '=', '+', '-', '*', '/', as well as LaTeX markers.
    
    if response == []:
        return False
    
    ## take the last element of the list (string response).
    ## avoid the case of small `x` in equation and `i`
    response = response[-1]
    if re.search(r'[\\=+\-*/><^]', response) or re.search(r'\bx\b', response) or re.search(r'\bi\b', response):
        return False

    # Convert text to uppercase for case-insensitive matching.
    text_upper = response.upper()
    # Extract candidate tokens that consist solely of the valid Roman numeral characters covering only I, V, X.
    tokens = re.findall(r'\b[XVI]+\b', text_upper)
    
    # Regex pattern for a valid Roman numeral.
    roman_pattern = re.compile(
        r"^M{0,3}(CM|CD|D?C{0,3})"
        r"(XC|XL|L?X{0,3})"
        r"(IX|IV|V?I{0,3})$"
    )
    
    for token in tokens:
        # Avoid misinterpreting an isolated "I" in normal text (e.g., as a pronoun),
        # unless the entire string is exactly "I".
        if token == "I" and text_upper.strip() != "I":
            continue
        if roman_pattern.fullmatch(token):
            return True
    return False


def convert_written_numbers(text: str) -> str:
    """
    Convert any written number words in the input text to their numeric equivalents.
    
    This function scans the input text (which can be plain text or embedded in LaTeX commands)
    and replaces any recognized written numbers (both cardinal and ordinal) with their corresponding digits.
    
    Examples:
      - "three" becomes "3"
      - "\\text{one and two}" becomes "\\text{1 and 2}"
    
    Parameters:
        text (str): The input string containing written number words.
    
    Returns:
        str: The modified text with all recognized written numbers replaced by numbers.
    """
    numwords = {
        # Cardinal numbers (units)
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        
        # Cardinal numbers (tens and compounds)
        "twenty": 20,
        "twenty-one": 21,
        "twenty one": 21,
        "twenty-two": 22,
        "twenty two": 22,
        "twenty-three": 23,
        "twenty three": 23,
        "twenty-four": 24,
        "twenty four": 24,
        "twenty-five": 25,
        "twenty five": 25,
        "twenty-six": 26,
        "twenty six": 26,
        "twenty-seven": 27,
        "twenty seven": 27,
        "twenty-eight": 28,
        "twenty eight": 28,
        "twenty-nine": 29,
        "twenty nine": 29,
        
        "thirty": 30,
        "thirty-one": 31,
        "thirty one": 31,
        "thirty-two": 32,
        "thirty two": 32,
        "thirty-three": 33,
        "thirty three": 33,
        "thirty-four": 34,
        "thirty four": 34,
        "thirty-five": 35,
        "thirty five": 35,
        "thirty-six": 36,
        "thirty six": 36,
        "thirty-seven": 37,
        "thirty seven": 37,
        "thirty-eight": 38,
        "thirty eight": 38,
        "thirty-nine": 39,
        "thirty nine": 39,
        
        "forty": 40,
        "forty-one": 41,
        "forty one": 41,
        "forty-two": 42,
        "forty two": 42,
        "forty-three": 43,
        "forty three": 43,
        "forty-four": 44,
        "forty four": 44,
        "forty-five": 45,
        "forty five": 45,
        "forty-six": 46,
        "forty six": 46,
        "forty-seven": 47,
        "forty seven": 47,
        "forty-eight": 48,
        "forty eight": 48,
        "forty-nine": 49,
        "forty nine": 49,
        
        # Scales
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
        "trillion": 1000000000000,
        
        # Ordinal numbers (explicit)
        "first": 1,
        "second": 2,
        "third": 3,
        "fifth": 5,
        "eighth": 8,
        "ninth": 9,
        "twelfth": 12,
        
        # Ordinal numbers (generated)
        "zeroth": 0,
        "onezeroth": 0,  # optional; typically "zeroth" is used
        "fourth": 4,
        "sixth": 6,
        "seventh": 7,
        "tenth": 10,
        "eleventh": 11,
        "thirteenth": 13,
        "fourteenth": 14,
        "fifteenth": 15,
        "sixteenth": 16,
        "seventeenth": 17,
        "eighteenth": 18,
        "nineteenth": 19,
        "twentieth": 20,
        "twenty-first": 21,
        "twenty first": 21,
        "twenty-second": 22,
        "twenty second": 22,
        "twenty-third": 23,
        "twenty third": 23,
        "twenty-fourth": 24,
        "twenty fourth": 24,
        "twenty-fifth": 25,
        "twenty fifth": 25,
        "twenty-sixth": 26,
        "twenty sixth": 26,
        "twenty-seventh": 27,
        "twenty seventh": 27,
        "twenty-eighth": 28,
        "twenty eighth": 28,
        "twenty-ninth": 29,
        "twenty ninth": 29,
        "thirtieth": 30,
        "thirty-first": 31,
        "thirty first": 31,
        "thirty-second": 32,
        "thirty second": 32,
        "thirty-third": 33,
        "thirty third": 33,
        "thirty-fourth": 34,
        "thirty fourth": 34,
        "thirty-fifth": 35,
        "thirty fifth": 35,
        "thirty-sixth": 36,
        "thirty sixth": 36,
        "thirty-seventh": 37,
        "thirty seventh": 37,
        "thirty-eighth": 38,
        "thirty eighth": 38,
        "thirty-ninth": 39,
        "thirty ninth": 39,
        "fortieth": 40,
        "forty-first": 41,
        "forty first": 41,
        "forty-second": 42,
        "forty second": 42,
        "forty-third": 43,
        "forty third": 43,
        "forty-fourth": 44,
        "forty fourth": 44,
        "forty-fifth": 45,
        "forty fifth": 45,
        "forty-sixth": 46,
        "forty sixth": 46,
        "forty-seventh": 47,
        "forty seventh": 47,
        "forty-eighth": 48,
        "forty eighth": 48,
        "forty-ninth": 49,
        "forty ninth": 49,
    }

    # Build a regular expression that matches any of the keys in numwords as whole words.
    pattern = r'\b(' + '|'.join(re.escape(word) for word in numwords.keys()) + r')\b'
    
    def repl(match: re.Match) -> str:
        word = match.group(0).lower()
        return str(numwords.get(word, word))
    
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def compare_percentage(gold, response):
    gold_float = float(gold[0])
    response_float = float(response[0])
    if gold_float == response_float:
        return True
    gold_scaled = gold_float * 100
    if gold_scaled == response_float:
        return True
    response_scaled = response_float * 100
    if gold_float == response_scaled:
        return True



def compare_percentage_2sf(gold, response):
    gold_float = float(gold[0])
    response_float = float(response[0])
    # Case 1: Scale gold_float by 100 (for cases where gold is in decimal but response is percentage)
    gold_scaled = gold_float * 100
    if abs(gold_scaled - response_float) < 1:
        return True
    
    # Case 2: Scale response_float by 100 (for cases where response is in decimal but gold is percentage)
    response_scaled = response_float * 100
    if abs(gold_float - response_scaled) < 1:
        return True
    
    # Case 3: No scaling (both are in same format)
    if abs(gold_float) < 1:
        relative_tolerance = 0.01  # 1% tolerance for 2 significant figures
        return abs((gold_float - response_float) / gold_float) < relative_tolerance
    else:
        return abs(gold_float - response_float) < 1

def verify_numeric_against_fraction(gold, response, is_percentage=False):
    """
    Verifies if a response is equivalent to a gold answer when one is a fraction and the other is a numeric type.
    
    This function handles cases where the gold answer is a numeric type (int or float) and the response
    is a Rational, or vice versa. It converts both to floats and compares them with appropriate precision
    based on the magnitude of the gold answer.

    Returns:
        bool: True if the response is equivalent to the gold answer within the specified tolerance, False otherwise.
              - For gold values < 1, uses a 1% relative tolerance (for approximately 2 significant figures)
              - For gold values >= 1, uses an absolute tolerance of 1
    """
    if (isinstance(response[0], Rational)) or \
       (isinstance(gold[0], Rational)):
        if is_percentage:
            return compare_percentage_2sf(gold, response)
        else:
            # Convert both to float and compare with appropriate precision
            gold_float = float(gold[0])
            response_float = float(response[0])
            if abs(gold_float) < 1:
                relative_tolerance = 0.01  # 1% tolerance for 2 significant figures
                return abs((gold_float - response_float) / gold_float) < relative_tolerance
            else:
                return abs(gold_float - response_float) < 1
    return False

def verify_complex_in_fraction(gold, response):
    """
    Verifies if a response is equivalent to a gold answer when dealing with complex numbers in fraction.
    Handles comparing both real and imaginary parts with appropriate tolerances.
    """
    
    # Get the first element (the expression) from each list
    gold_expr = gold[0]
    response_expr = response[0]
    
    # Check if either expression contains the symbol 'i' for complex numbers
    from sympy import Symbol, I, re, im, sympify, Float
    i_symbol = Symbol('i', real=True)
    
    gold_has_i = False
    response_has_i = False
    
    # Check if gold contains 'i'
    if hasattr(gold_expr, 'free_symbols') and i_symbol in gold_expr.free_symbols:
        gold_has_i = True
        
    # Check if response contains 'i'
    if hasattr(response_expr, 'free_symbols') and i_symbol in response_expr.free_symbols:
        response_has_i = True
    
    # If neither has complex component, no need for complex comparison
    if not (gold_has_i or response_has_i):
        return False
    
    try:
        # Try to convert expressions to sympy expressions if they're not already
        if not isinstance(gold_expr, (Float, int, complex)):
            try:
                gold_expr = sympify(str(gold_expr).replace('i', 'I'))
            except:
                pass
                
        if not isinstance(response_expr, (Float, int, complex)):
            try:
                response_expr = sympify(str(response_expr).replace('i', 'I'))
            except:
                pass
        
        # Evaluate both expressions
        gold_eval = gold_expr.evalf()
        response_eval = response_expr.evalf()
        
        # Extract real and imaginary parts
        gold_real = float(re(gold_eval))
        gold_imag = float(im(gold_eval))
        response_real = float(re(response_eval))
        response_imag = float(im(response_eval))
        
        # Apply appropriate tolerance based on magnitude
        # For real part
        if abs(gold_real) < 1:
            real_tolerance = abs(gold_real) * 0.01  # 1% relative tolerance for 2 significant figures
        else:
            real_tolerance = 1  # Absolute tolerance of 1 for values >= 1
            
        # For imaginary part
        if abs(gold_imag) < 1:
            imag_tolerance = abs(gold_imag) * 0.01  # 1% relative tolerance for 2 significant figures
        else:
            imag_tolerance = 1  # Absolute tolerance of 1 for values >= 1
        
        # Compare both parts within tolerance
        real_match = abs(gold_real - response_real) <= real_tolerance
        imag_match = abs(gold_imag - response_imag) <= imag_tolerance
        
        return real_match and imag_match
        
    except Exception as e:
        print(f"Error processing complex numbers: {e}")
        return False


def main():
    import json
    import os
    
    # Load the dataset
    with open("tocheck_w_cleaned_gold.json", "r") as f:
        data = json.load(f)
    
    total_correct = 0
    total_questions = len(data)
    incorrect_cases = []
    
    for i, item in enumerate(data):
        question_id = item["id"]
        response = item["response_last_300charactors"]
        ground_truth = item["cleaned_gold"]
        
        score = compute_score(response, ground_truth)
        
        if score > 0:
            total_correct += 1
            result = "CORRECT"
        else:
            result = "INCORRECT"
            incorrect_cases.append({
                "id": question_id,
                "question": item["question"],
                "response": response,
                "ground_truth": ground_truth
            })
            
        print(f"Question {i+1}/{total_questions} (ID: {question_id}): {result}")
    
    accuracy = (total_correct / total_questions) * 100
    print(f"\nTotal Score: {total_correct}/{total_questions} ({accuracy:.2f}%)")
    
    # Write incorrect cases to JSON file
    if incorrect_cases:
        with open("incorrect_cases.json", "w", encoding="utf-8") as jsonfile:
            json.dump(incorrect_cases, jsonfile, indent=4, ensure_ascii=False)
        
        print(f"\nIncorrect cases written to incorrect_cases.json")

if __name__ == "__main__":
    main()
    #print(fraction_match_cases)