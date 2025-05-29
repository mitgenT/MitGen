'''
This program performs code generation.
'''
import ast
from datetime import date, timedelta
from func_timeout import FunctionTimedOut, func_set_timeout
import inspect
from pprint import pprint
import random
from openai import AzureOpenAI,OpenAI
from http import HTTPStatus
import dashscope
from dashscope import Generation  # 建议dashscope SDK 的版本 >= 1.14.0
# import openai
import logging
import os
import sys
# sys.path.append('/data/toli/State-Level-DP/mutation')
import mutation.api
from mutation import utils
from mutation.perform_mutation import ast_mutation
from mutation.utils import initialisation, list_txt_files, file_check, root, is_same_syntax_by_ast, \
    wrap_as_function, analyze_indentation, list_txt_files_without_subdir, diff, Identifier, get_indent, \
    get_line_num, remove_empty_line_and_comment_from_code, load_args, get_target_code_path, get_class_name
import time
import json
import requests
import traceback 
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import torch
from typing import List, Optional

sys.path.append("/data/toli/State-Level-DP/mutation")
sys.path.append("/data/toli/llama3")
from llama import Dialog, Llama


def extract_code(generated_code: str):
    # print("Extracting code...")
    extracted_lines = []
    code_scope = False
    has_code_backquote = False
    for each_line in generated_code.splitlines():
        if "<MASK>" in each_line:
            continue
        if "```" in each_line:
            if not code_scope:
                code_scope = True
                has_code_backquote = True
                continue
            else:
                break
        if code_scope:
            extracted_lines.append(each_line)
    if not has_code_backquote:
        return generated_code
    return "\n".join(extracted_lines)


# This is a chat completion for OpenAI Python library version < 1.0.0
# def prompt_chatgpt(concatenated_content: str):
#     openai.api_base = "https://hkust.azure-api.net"
#     openai.api_key = chatgpt_api_key
#     openai.api_type = "azure"
#     openai.api_version = "2023-12-01-preview"
#
#     system_msg = 'You are a helpful assistant.'
#     try:
#         response = openai.ChatCompletion.create(engine="gpt-35-turbo",
#                                                 temperature=1.0,
#                                                 messages=[
#                                                     {"role": "system", "content": system_msg},
#                                                     {"role": "user", "content": concatenated_content}
#                                                 ])
#     except Exception as e:
#         print(f"Encounter {e}, sleep for 10 seconds now.")
#         time.sleep(10)
#         return None
#     output_code = response["choices"][0]["message"]["content"]
#     return output_code


def get_class_function_name(code):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    result.append(child.name)
            return result


def prompt_ali(concatenated_content: str, input_temperature:int=1.0):
    dashscope.api_key = mutation.api.qwen_api_key
    messages = [{'role': 'system', 'content': 'You are a helpful coding assistant.'},
                {'role': 'user', 'content': concatenated_content}]
    response = None
    try:
        response = Generation.call(model="qwen-turbo",
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                seed=random.randint(1, 10000),
                                temperature=input_temperature,
                                # 将输出设置为"message"格式
                                result_format='message')
    except Exception as e:
        print(f"Encounter {e}, sleep for 10 seconds now.")
        time.sleep(10)
        return None
    # return output_code
    if response.status_code == HTTPStatus.OK:
        return response["output"]["choices"][0]["message"]["content"]
    else:
        return None


# This is a new chat completion for OpenAI Python 1.x
def prompt_chatgpt(concatenated_content: str, input_temperature:int=1.0):
    client = AzureOpenAI(
        azure_endpoint="https://hkust.azure-api.net",
        api_key=mutation.api.chatgpt_api_key,
        api_version="2023-12-01-preview"
    )

    system_msg = 'You are a helpful assistant.'
    try:
        response = client.chat.completions.create(model="gpt-35-turbo",
                                                temperature=input_temperature,
                                                messages=[
                                                    {"role": "system", "content": system_msg},
                                                    {"role": "user", "content": concatenated_content}
                                                ])
        # print(f"concatenated_content:{concatenated_content}")
    except Exception as e:
        print(f"Encounter {e}, sleep for 10 seconds now.")
        time.sleep(10)
        return None
    output_code = response.choices[0].message.content
    return output_code


def prompt_gemini(concatenated_content: str, input_temperature:int=1.0):
    url = f'https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=AIzaSyCl_qxBGn1RULFXrQgvfTHVsO7iUwuJapU'
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [{"text": concatenated_content}]
            }
        ],
        "generationConfig": {
            "temperature": input_temperature
        }

    }
    response = None
    try:
        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(f"Get exception: {e}")
        print("Encounter server overloaded error, sleep for 10 seconds now.")
        time.sleep(10)
        return None
    # print(f"response status_code: {response.status_code}")
    # print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    output_code = None
    try:
        output_code = \
            json.loads(json.dumps(response.json(), indent=4, ensure_ascii=False))['candidates'][0]['content']['parts'][
                0][
                'text']
    except Exception:
        if response.reason == "OK":
            print(f"Get exception: {response}")
        else:
            print(f"Get exception: {response.reason}")
        return None
    return output_code


def prompt_ali_llama(concatenated_content: str, input_temperature:int=2.0):
    dashscope.api_key = "sk-be9281631e604916bf78154e4c0c8ae6"
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': concatenated_content}]
    response = dashscope.Generation.call(
        model='llama3-70b-instruct',
        messages=messages,
        seed=random.randint(1, 1000000),
        temperature=input_temperature,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        if response.status_code == 429:
            time.sleep(50)
        time.sleep(10)
        return None


def is_full_code(code, original_code,mask_length):
    #mask_length_arg is for debugging
    # if mask_length_debug_arg != None:
    # mask_length = mask_length_debug_arg
    code_length = len(code.splitlines())
    #dist_to_mask = extracted code's length - mask location's length
    dist_to_mask = abs(code_length - mask_length)
    #dist_to_origin = original code's length - extracted code's length
    dist_to_origin = abs(len(original_code.splitlines()) - code_length)
    return dist_to_origin < dist_to_mask


def read_docstring(target):
    with open(f"{root}/prompt/{target}", 'r', encoding="utf-8") as file1:
        return file1.read()


# def refine_extracted_code_for_mask_location_match_overlapped(masked_code_lines:list[str], extracted_code_lines:list[str]):
#     stripped_masked_code_lines = []
#     for idx, each_line in enumerate(masked_code_lines):
#         if each_line:
#             stripped_masked_code_lines.append((idx,each_line))
    
#     stripped_extracted_code_lines = []
#     for idx, each_line in enumerate(extracted_code_lines):
#         if each_line:
#             stripped_extracted_code_lines.append((idx,each_line))

#     matched_dict = {}
#     for each_masked_code_tuple in stripped_masked_code_lines:
#         for each_extracted_code_tuple in stripped_extracted_code_lines:
#             if each_masked_code_tuple[0] == each_extracted_code_tuple[0]:


# Handle the case that generated code contains lines exists in the original code, in this case, the redundant lines in generated code should be pruned.
# Note that this function only matches consecutive block
def refine_extracted_code_for_mask_location(masked_code_lines: list[str], extracted_code_lines: list[str]):
    # print(f"Refining code for mask location.")

    if len(extracted_code_lines) == 0:
        return None

    # closest_uppest_matched_line is the line of CUT that matches the line of extracted_code_lines and closes to function signature
    closest_uppest_matched_extracted_code_line = -1
    # closest_upppest_matched_line is the matched line of CUT furthest from the function signature
    closest_lowest_matched_extracted_code_line = len(extracted_code_lines)

    # tmp_matched_extracted_code_lines_idx store uppermost's line of extracted code that is the same as CUT, this allow us to prune generated code's lines prior to matched_extracted_code_lines_idx
    tmp_matched_extracted_code_lines_idx = -1
    # uppermost_matched_extracted_code_line is the greatest matched_extracted_code_lines_idx in forard search
    # uppermost_matched_extracted_code_line = -1
    # print(f"masked_code_lines:{"\n".join(masked_code_lines)}")
    # print(f"extracted_code_lines:{"\n".join(extracted_code_lines)}")
    closest_uppest_matched_extracted_code_line_to_line_count_dict = {}
    #count the number of matched line for a closest_lowest_matched_extracted_code_line value
    line_count = 0
    for each_line_idx in range(len(masked_code_lines)):
        #ignore blank line
        if masked_code_lines[each_line_idx].strip() == '':
            continue
        # When hitting "<MASK>", flush the search result
        if "<MASK>" in masked_code_lines[each_line_idx]:
            # print(f"!!!!Hitting upper mask!!!!!")
            if line_count != 0:
                closest_uppest_matched_extracted_code_line_to_line_count_dict[
                    tmp_matched_extracted_code_lines_idx] = line_count
            if closest_uppest_matched_extracted_code_line_to_line_count_dict:
                closest_uppest_matched_extracted_code_line = \
                sorted(closest_uppest_matched_extracted_code_line_to_line_count_dict.items(),
                       key=lambda item: (-item[1], -item[0]))[0][0]
            else:
                closest_uppest_matched_extracted_code_line = -1
            # print(f"sorted closest_uppest_matched_extracted_code_line_to_line_count_dict:{sorted(closest_uppest_matched_extracted_code_line_to_line_count_dict.items(), key=lambda item: (-item[1], -item[0]))}")
            # print(f"closest_uppest_matched_extracted_code_line:{closest_uppest_matched_extracted_code_line}")
            break

        # Meaning: if the line in CUT is the same as the line in generated code snippet
        # print(f"masked_code_lines[each_line_idx].strip():{masked_code_lines[each_line_idx].strip()}")
        # print(f"extracted_code_lines:{extracted_code_lines}")
        # print(f"tmp_matched_extracted_code_lines_idx:{tmp_matched_extracted_code_lines_idx}")
        # print(f"extracted_code_lines[tmp_matched_extracted_code_lines_idx]:{extracted_code_lines[tmp_matched_extracted_code_lines_idx]}")
        if masked_code_lines[each_line_idx].strip() == extracted_code_lines[
            tmp_matched_extracted_code_lines_idx + 1].strip():

            # print(f"!!!tmp_matched_extracted_code_lines_idx:{tmp_matched_extracted_code_lines_idx}")
            # print(f"!!!masked_code_lines[each_line_idx]:{masked_code_lines[each_line_idx]}")
            # print(f"!!!extracted_code_lines:{extracted_code_lines[tmp_matched_extracted_code_lines_idx + 1]}")
            tmp_matched_extracted_code_lines_idx += 1
            line_count += 1
            # if tmp_matched_extracted_code_lines_idx > uppermost_matched_extracted_code_line:
            #     uppermost_matched_extracted_code_line = tmp_matched_extracted_code_lines_idx

            # Handle the case that extracted_code is completely redundant with code under test, at this case, just need to return empty string
            if tmp_matched_extracted_code_lines_idx == len(extracted_code_lines) - 1:
                return None
        else:
            # print(f"!!!tmp_matched_extracted_code_lines_idx:{tmp_matched_extracted_code_lines_idx}")
            # print(f"!!!masked_code_lines[each_line_idx]:{masked_code_lines[each_line_idx]}")
            # print(f"!!!extracted_code_lines:{extracted_code_lines[tmp_matched_extracted_code_lines_idx + 1]}")
            # When there is a line which the extracted line does not match cut before hitting <MASK>, the lines are just conincidentally matched (i.e., lines matched outside the mask location), hence tmp_uppest_matched_extracted_code_line is reset
            if line_count != 0:
                closest_uppest_matched_extracted_code_line_to_line_count_dict[tmp_matched_extracted_code_lines_idx] = line_count
            line_count = 0
            tmp_matched_extracted_code_lines_idx = -1

    # print(f"!!!closest_uppest_matched_extracted_code_line:{closest_uppest_matched_extracted_code_line}")

    # Compute closest_lowest_matched_line
    tmp_matched_extracted_code_lines_idx = len(extracted_code_lines)
    # lowermost_matched_extracted_code_line = len(extracted_code_lines)
    # First reverse extracted code
    # reversed_extracted_code_lines = list(reversed(extracted_code_lines))
    closest_lowest_matched_extracted_code_line_to_line_count_dict = {}
    line_count = 0

    #Handle the case that both masked code and extracted code has a return statemeent in the last line
    # if "return " in masked_code_lines[-1]

    #A variable to help checking whether the last line of extracted code and masked code are return statement
    last_masked_code_line_has_been_explored = False



    for idx, each_line_idx in enumerate(list(reversed(range(len(masked_code_lines))))):
        # print(f"reversed idx:{each_line_idx}")
        #ignore blank line
        if masked_code_lines[each_line_idx].strip() == '':
            continue
        #If the last line of both masked and extracted code have return statement, can directly flush the result
        # if last_masked_code_line_has_been_explored == False and "return " in masked_code_lines[each_line_idx] and "return " in extracted_code_lines[-1]:
        #     closest_lowest_matched_extracted_code_line = tmp_matched_extracted_code_lines_idx
        #     break

        # When hitting "<MASK>", flush the search result
        if "<MASK>" in masked_code_lines[each_line_idx]:
            # print(f"!!!!Hitting lower mask!!!!!")
            if line_count != 0:
                closest_lowest_matched_extracted_code_line_to_line_count_dict[
                    tmp_matched_extracted_code_lines_idx] = line_count
            if closest_lowest_matched_extracted_code_line_to_line_count_dict:
                closest_lowest_matched_extracted_code_line = \
                sorted(closest_lowest_matched_extracted_code_line_to_line_count_dict.items(),
                       key=lambda item: (-item[1], item[0]))[0][0]
            else:
                closest_lowest_matched_extracted_code_line = len(extracted_code_lines)
            # print(f"closest_lowest_matched_extracted_code_line:{closest_lowest_matched_extracted_code_line}")
            break

        if masked_code_lines[each_line_idx].strip() == extracted_code_lines[
            tmp_matched_extracted_code_lines_idx - 1].strip():
            # print(f"!!!tmp_matched_extracted_code_lines_idx:{tmp_matched_extracted_code_lines_idx}")
            # print(f"!!!masked_code_lines[each_line_idx]:{masked_code_lines[each_line_idx]}")
            # print(f"!!!extracted_code_lines:{extracted_code_lines[tmp_matched_extracted_code_lines_idx-1]}")
            tmp_matched_extracted_code_lines_idx -= 1
            line_count += 1
            last_masked_code_line_has_been_explored = True

            # if tmp_matched_extracted_code_lines_idx < lowermost_matched_extracted_code_line:
            #     #Note that it is impossible for closest_uppest_matched_line == closest_lowest_matched_line, because the first loop already handles the case that "extracted_code is completely redunadant"
            #     lowermost_matched_extracted_code_line = tmp_matched_extracted_code_lines_idx

        else: #ifmasked_code_lines[each_line_idx].strip() and extracted_code_lines[ tmp_matched_extracted_code_lines_idx - 1] are not matched
            if line_count != 0:
                closest_lowest_matched_extracted_code_line_to_line_count_dict[
                    tmp_matched_extracted_code_lines_idx] = line_count
            line_count = 0
            tmp_matched_extracted_code_lines_idx = len(extracted_code_lines)
            # last_masked_code_line_has_been_explored = True

    new_extracted_code = []
    # if closest_uppest_matched_extracted_code_line != -1 or closest_lowest_matched_extracted_code_line != len(extracted_code_lines):
    # Note that closest_lowest_matched_CUT_line - closest_uppest_matched_CUT_line >= 2 for sure, as the first loop already handles "extracted_code is completely redunadant" (i.e., the difference of the two <= 1)
    # print(f"closest_uppest_matched_extracted_code_line:{closest_uppest_matched_extracted_code_line}")
    # print(f"closest_lowest_matched_extracted_code_line:{closest_lowest_matched_extracted_code_line}")
    for line_idx in range(closest_uppest_matched_extracted_code_line + 1, closest_lowest_matched_extracted_code_line):
        new_extracted_code.append(extracted_code_lines[line_idx])
    if new_extracted_code == []:
        return ""
    else:
        return "\n".join(new_extracted_code)


# def remove_docstring_from_generated_code(extracted_code_lines):
#     docstring = open("mutation/subjects/prompt/evo1.txt","r")
#     for each_line in extracted_code_lines:


def snippet_infilling(masked_code:str, extracted_code:str):
    is_enable_template = False
    masked_code_lines = masked_code.splitlines()
    indent = ""

    # print(f"extracted_code:\n{extracted_code}")

    #Handle the case that extracted_code code equals None (i.e., completely overlapped with masked code)
    if extracted_code == None:
        resultant_code = []
        for index, each_line in enumerate(masked_code.splitlines()):
            if "<MASK>" not in each_line:
                resultant_code.append(each_line)
        return "\n".join(resultant_code)

    line_to_mask = None
    for each_line in masked_code_lines:
        if "<MASK>" in each_line:
            line_to_mask = each_line
            if each_line.strip() != "<MASK>":
                #To enter this loop, the line must contain code other than <MASK> (e.g., if), in this case, a template must be used.
                is_enable_template = True
            indent = get_indent(each_line)
            break

    raw_extracted_code_lines: list[str] = extracted_code.splitlines()

    #infilling_mode_enum is a label indicating refinement needed for infilling generated code.
    #directly_replace means generated code snippet is fine, no action is needed.
    infilling_mode_enum = ["directly_replace", "add_indent", "no_need_indent", "need_to_adapt_indent", "", "remove_docstring"]
    # infilling_mode = infilling_mode_enum[0]
    extracted_code_lines = []

    #Remove empty lines from extracted code
    for code_line in raw_extracted_code_lines:
        if code_line.strip() != "":
            extracted_code_lines.append(code_line)

    #block_indent_diff computes the adjustment in indentation of generated code, based on the indentation of the first line 
    block_first_line_indent = None
    for index, code_line in enumerate(extracted_code_lines):
        
        #Note that the index 0 is the first line of extracted code, not complete code
        if index == 0:
            block_first_line_indent = len(get_indent(code_line))
            #Check whether identation of extracted code is the same as the one of mask location
            # if utils.get_indent(code_line) == indent:
                # #infilling_mode = no_need_indent, which means no modification on the indentation is needed.
                # infilling_mode = infilling_mode_enum[2] 
                # extracted_code_lines[index] = indent + code_line.strip()

                # #is enable template is for generating diverse code
                # if is_enable_template:
                #     extracted_code_lines[index] = indent + extracted_code_lines[index]
            # else:
                # infilling_mode = infilling_mode_enum[3]
                # block_first_line_indent = len(utils.get_indent(code_line))
                # block_first_line_indent = len(utils.get_indent(code_line)) - len(indent)
                # indent_to_reduce = len(utils.get_indent(code_line))
            # elif len(utils.get_indent(code_line)) > len(indent):
            #     infilling_mode = infilling_mode_enum[3]
            # else:
            #     infilling_mode = infilling_mode_enum[3]
        # if infilling_mode == infilling_mode_enum[3]:
        if len(get_indent(code_line)) > block_first_line_indent:
            indent_to_add_back = len(get_indent(code_line)) - block_first_line_indent
            extracted_code_lines[index] = indent + "".join([" " for i in range(indent_to_add_back)]) + code_line.strip()
        elif len(get_indent(code_line)) < block_first_line_indent:
            indent_to_reduce_back = min(block_first_line_indent - len(get_indent(code_line)),len(indent))
            extracted_code_lines[index] = indent[:indent_to_reduce_back] + code_line.strip()
        else:
            extracted_code_lines[index] = indent + code_line.strip()
        # #Check the second line (e.g., line after predicate)
        # elif index == 1 and infilling_mode != infilling_mode_enum[2] and infilling_mode != infilling_mode_enum[3]:
        #     # print(f"!!!What is extracted_code_lines[0][-1]:{extracted_code_lines[0][-1]}")
        #     if extracted_code_lines[0][-1] != ":":
                
        #         #Second line has more indents than the first line
        #         if len(utils.get_indent(extracted_code_lines[index])) > len(utils.get_indent(extracted_code_lines[0])):
        #             infilling_mode = infilling_mode_enum[0]
        #         else:
        #             print(f"1111111.assigning infilling_mode_enum[1] for {extracted_code_lines[index]}")
        #             infilling_mode = infilling_mode_enum[1]
        #     else:
        #         #The case that the discrepancy in indentation is larger than 4.
        #         # print(f"!!!diff in indent:{len(utils.get_indent(extracted_code_lines[index])) - len(utils.get_indent(extracted_code_lines[0]))}")
        #         if len(utils.get_indent(extracted_code_lines[index])) - len(utils.get_indent(extracted_code_lines[0])) > 4:
        #             infilling_mode = infilling_mode_enum[0]
        #         else:
        #             print(f"2222222.assigning infilling_mode_enum[1] for {extracted_code_lines[index]}")
        #             infilling_mode = infilling_mode_enum[1]


        # if infilling_mode == infilling_mode_enum[1]:
        #     print(f"Entering infilling_mode_enum[1\], indent is {indent}, code_line is {code_line}")
        #     extracted_code_lines[index] = indent + code_line
        # elif infilling_mode == infilling_mode_enum[3]:
        #     if block_first_line_indent
            # dummy = 1
            # #remove indents if possible
            # if len(utils.get_indent(code_line)) >= indent_to_reduce:
            #     extracted_code_lines[index] = code_line[abs(indent_to_reduce):]
            # else:
            #     #Indent that should be removed is greater than the indent that the line has, in this case, there should be a bug, but we handle this case in best effort manner.
            #     extracted_code_lines[index] = code_line.strip()

            # print(f"Entering infilling_mode_enum[3], indent is {indent}, code_line is {code_line}")
            # print(f"block_indent_diff:{block_indent_diff}")
            
            #indent of extracted code is greater than mask code
            # if indent_to_reduce > 0:
            #     print(f"Before adjustment:{code_line}")
            #     extracted_code_lines[index] = code_line[abs(indent_to_reduce):]
            #     print(f"After adjustment:{extracted_code_lines[index]}")
            # else:
            #     extracted_code_lines[index] = "".join([" " for i in range(indent_to_reduce)]) + code_line
    
    if is_enable_template:
        mask_start_index = 0
        for index in range(len(masked_code_lines) - 1, -1, -1):
            each_line = masked_code_lines[index]
            if "<MASK>" in each_line:
                del masked_code_lines[index]
                mask_start_index = index
        masked_code_lines = masked_code_lines[:mask_start_index] + extracted_code_lines + masked_code_lines[mask_start_index:]
        # print(f"Code to return2")
        # print("\n".join(masked_code_lines))
        return "\n".join(masked_code_lines)
    # print(f"Code to return:{masked_code.replace(line_to_mask, "\n".join(extracted_code_lines))}")
    return masked_code.replace(line_to_mask, "\n".join(extracted_code_lines))


def docstring_convention(docstring):
    result = docstring.replace("—", "-")
    result = result.replace("𝑎", "a")
    result = result.replace("𝑏", "b")
    result = result.replace("𝑖", "i")
    result = result.replace("𝑗", "j")
    result = result.replace("𝑘", "k")
    result = result.replace("𝑛", "n")
    result = result.replace("𝑡", "t")
    result = result.replace("≤", "<=")
    return result


def get_test_input(subject, model):
    # assert os.path.isfile(path)
    # print(f"!!!!Path:{root}/test_input/{subject}/{model}")
    test_input_list = list_txt_files(f"{root}/test_input/{subject}/{model}")

    # test_input_list = f"/data/toli/State-Level-DP/mutation/subjects/test_input/classeval_accessgatewayfilter.txt"
    assert len(test_input_list) > 0, f"Cannot find any test input, please check whether they exist under {root}/test_input/{subject}/{model}"
    # print(f"test_input_list:{test_input_list}")
    with open(test_input_list[0], "r", encoding="utf-8") as f:
        test_input = f.read()
    return test_input


def pass_test_case(func, args=""):
    match args:
        case "":
            return eval(f"{func}()")
        case _:
            return eval(f"{func}({args})")


class GenerateCodeWithLLM:
    def __init__(self, target: str, model: str, mode: str, test_mode, input_mode, dp_mode, prioritization=False, mask_location=None):
        self.identifier: Identifier = None
        self.input_mode = input_mode
        self.input_mode_class = None
        self.dp_mode = dp_mode
        if prioritization == False:
            self.start_no, self.end_no = self.dp_mode.get_start_end()
        else:
            self.start_no, self.end_no = self.dp_mode.get_start_end()
            self.end_no = 1
        self.mask_location = mask_location
        self.prioritization = prioritization
        self.retry_time = 0
        self.syntax_retry_time = 0
        self.model = model
        self.mode = mode
        self.mode_class: GenerateCodeBySingle | GenerateCodeByAst = None
        self.test_mode = test_mode
        self.tokenizer = None
        self.model_loader = None
        self.pipe = None
        self.output_code_path = None
        self.mask_length = None
        self.subject = None
        self.update_target(target)
        os.makedirs(f"mutation/subjects/test_input/{self.subject}/{model}", exist_ok=True)

    def update_target(self,target:str):
        self.target = target
        if not target == None:
            self.subject = target.removesuffix(".txt")

    def initialise_hf_model(self):
        model_dict = {"deepseek33":"deepseek-ai/deepseek-coder-33b-instruct",
                      "deepseek67":"deepseek-ai/deepseek-coder-6.7b-instruct",
                      "deepseek13":"deepseek-ai/deepseek-coder-1.3b-instruct",
                      "deepseekv2":"deepseek-ai/DeepSeek-V2-Lite-Chat",
                      "deepseekr1_14":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                      "deepseekr1_32":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
        if "deepseek" in self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dict[self.model]}", trust_remote_code=True)
            self.model_loader = AutoModelForCausalLM.from_pretrained(f"{model_dict[self.model]}", trust_remote_code=True, torch_dtype=torch.bfloat16,device_map='auto')
            if "deepseekv2" in self.model:
                self.model_loader.generation_config = GenerationConfig.from_pretrained(f"{model_dict[self.model]}")
                self.model_loader.generation_config.pad_token_id = self.model_loader.generation_config.eos_token_id
        elif "qwenv2" in self.model:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
            self.model_loader = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                torch_dtype="auto",
                device_map="auto"
            )
        elif "starchat" in self.model:
            self.pipe = pipeline(
                "text-generation",
                model="HuggingFaceH4/starchat2-15b-v0.1",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

    def check_exception(self):
        with open(f"{root}/correct_code/{self.input_mode_class.subject}.txt", "r", encoding="utf-8") as f:
            correct_code = f.read()
        if not self.verifying_complete_code(correct_code):
            raise SyntaxError("Correct code raise Exception, check bugs manually.")
            
    def check_if_generated_before(self, input_file_path: str, gen_index=None):
        output_file = ""
        if self.output_code_path == None:
            if gen_index is not None:
                output_file = self.creating_output_code_path(input_file_path, gen_index, self.prioritization)
            else:
                for index in range(self.start_no, self.end_no + 1):
                    output_file = self.creating_output_code_path(input_file_path, index, self.prioritization)
        else:
            output_file = self.output_code_path

        # print(f"Checking existence of {output_file}")
        if not os.path.isfile(output_file):
            return False
        else:
            return True

    def creating_output_code_path(self, path_to_code_to_be_infilled: str, ith_code_generated: int, prioritisation=False, docstring_verifier=False,generated_docstring_index=None):
        '''
        :return: "mutation/subjects/output/c1/ast/subject/model/xxx_ith_code_generated.txt"
        '''
        assert not (prioritisation and docstring_verifier), "prioritisation and docstring_verifier should be not set to True at the same time."
        if "test_input" in path_to_code_to_be_infilled:
            output_file_path = path_to_code_to_be_infilled.replace(".txt", f"_{ith_code_generated}.txt")
        else:
            if prioritisation == True:
                output_file_path = (path_to_code_to_be_infilled.replace("/input/", "/output/")
                    .replace("/simply_replace/","/prioritization/")
                    .replace(f"/{self.subject}/", f"/{self.subject}/{self.model}/")
                    .replace(".txt", f"_{ith_code_generated}.txt"))
                # print(f"!!!!output_file:{output_file}")
                os.makedirs('/'.join(output_file_path.split("/")[:-1]), exist_ok=True)
                os.makedirs('/'.join(output_file_path.split("/")[:-1]) + "/raw_output/", exist_ok=True)
            elif docstring_verifier == True:
                # assert len(str(generated_docstring_index)) == 1, "only single digit is allowed"
                # print(f"!!!self.subject is {self.subject}")
                # print(f"!!!path_to_code_to_be_infilled is {path_to_code_to_be_infilled}")
                # print(f"!!!path_to_code_to_be_infilled is {path_to_code_to_be_infilled}")
                # print(f"!!!self.subject is {self.subject}")
                # print(f"!!!/{self.subject}/")
                #!!!remember the last item in the path is "/evo14", not "/evo14/"
                dirname = os.path.dirname(path_to_code_to_be_infilled).replace("/input/", "/output/").replace("/ast/","/docstring_verifier/").replace("/simply_replace/","/docstring_verifier/").replace(f"/{self.subject}", f"/{self.subject}/{self.model}/{generated_docstring_index}/")
                # dirname = os.path.dirname(path_to_code_to_be_infilled).replace(f"/evo14", "123")
                # evco14_in_name = "/evo14/" in path_to_code_to_be_infilled
                # print(f"evco14_in_name:{evco14_in_name}")
                # print(f"!!!input path is {path_to_code_to_be_infilled}")
                # print(f"!!!dirname is {dirname}")
                basename = f"dv_{generated_docstring_index}_" + os.path.basename(path_to_code_to_be_infilled).replace(".txt", f"_{ith_code_generated}.txt")
                output_file_path = os.path.join(dirname,basename)
                # path_to_code_to_be_infilled.replace("/input/", "/output/")
                #     .replace("/simply_replace/","/docstring_verifier/")
                #     .replace(f"/{self.subject}/", f"/{self.subject}/{self.model}/{generated_docstring_index}")

                # print(f"!!!!output_file:{output_file}")
                os.makedirs('/'.join(output_file_path.split("/")[:-1]), exist_ok=True)
                os.makedirs('/'.join(output_file_path.split("/")[:-1]) + "/docstring/", exist_ok=True)
                os.makedirs('/'.join(output_file_path.split("/")[:-1]), exist_ok=True)
                # os.makedirs('/'.join(output_file_path.split("/")[:-1]) + "/raw_output/", exist_ok=True)
                os.makedirs('/'.join(output_file_path.split("/")[:-1]) + "/docstring/raw_output/", exist_ok=True)
                os.makedirs('/'.join(output_file_path.split("/")[:-1]) + "/raw_output/", exist_ok=True)
            else:
                output_file_path = (path_to_code_to_be_infilled.replace("/input/", "/output/")
                               .replace(f"/{self.subject}/", f"/{self.subject}/{self.model}/")
                               .replace(".txt", f"_{ith_code_generated}.txt"))
        return output_file_path

    def generate_test_input(self, target_code_path, concatenated_content, output_path):
        with open(f'{root}/prompt/example_for_input_{self.input_mode}.txt', encoding="utf-8") as f:
            example_content = f.read()
        concatenated_content = f"{example_content}\n\n{concatenated_content}"
        for ith_code_generated in range(1, 11):
            if self.check_if_generated_before(output_path, ith_code_generated):
                continue
            print(f"Generating {ith_code_generated}th test input")
            generated_complete_code = False
            trial = 0
            start_time = time.time()
            while not generated_complete_code:
                if time.time() - start_time > 120:
                    break
                output_code = self.prompt_llm(concatenated_content)
                if output_code == None:
                    continue
                generated_complete_code = True
                extracted_code = extract_code(output_code)
                print(f"Trial {trial}: verifying...")
                trial += 1
                if self.input_mode == "console":
                    subp = subprocess.Popen(["python", target_code_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                    out, err = subp.communicate(input=extracted_code.encode())
                    if err != b'':
                        print(f"error for generating test input:{err}")
                        generated_complete_code = False
                else:
                    with open(target_code_path, encoding="utf-8") as f:
                        CUT = f.read()
                    try:
                        result = self.verify_test_case(CUT, extracted_code)
                        if not result:
                            print(f"Not passing verification")
                            generated_complete_code = False
                            continue
                        elif self.input_mode == "function":
                            extracted_code = "|".join(result)
                    except FunctionTimedOut:
                        break
                if generated_complete_code:
                    output_code_path = self.creating_output_code_path(output_path, ith_code_generated)
                    print(f"Test input output path:{output_code_path}")
                    extracted_code_lines = []
                    for test_case_line in extracted_code.splitlines():
                        for test_case in test_case_line.split('|'):
                            if not test_case.startswith('#'):
                                extracted_code_lines.append(test_case)
                    rebuild_inputs = "\n".join(extracted_code_lines)
                    with open(output_code_path, "w", encoding="utf-8") as output_file:
                        output_file.write(rebuild_inputs)

    def identical_syntax_list(self, masked_file_path):
        """
        :param masked_file_path: One masked file path.
        :return: List of removed same syntax files.
        """
        masked_file_path_without_extension = masked_file_path.split("/")[-1].removesuffix(".txt")
        output_file_path = f"{root}/output/c1/ast/{self.subject}/{self.model}"
        output_file_path_list = list_txt_files_without_subdir(output_file_path)
        deleted_files = []
        identical_syntax = []
        divided_group = []
        for c1_file_name in output_file_path_list:
            if masked_file_path_without_extension in c1_file_name:
                divided_group.append(c1_file_name)
        for i in range(len(divided_group)):
            file1 = divided_group[i]
            for index2 in range(i + 1, len(divided_group)):
                file2 = divided_group[index2]
                with open(file1, encoding="utf-8") as f:
                    file1_text = f.read()
                with open(file2, encoding="utf-8") as f:
                    file2_text = f.read()
                if is_same_syntax_by_ast(ast.parse(file1_text), ast.parse(file2_text)):
                    identical_syntax.append([file1, file2])
        for pair in identical_syntax:
            try:
                os.remove(pair[1])
                deleted_files.append(pair[1])
            except Exception:
                pass
        deleted_files.sort()
        return deleted_files

    def is_identical_code(self, output_code,output_code_path):
        output_filepath_list = list_txt_files_without_subdir(os.path.dirname(output_code_path))
        # print(f"output_filepath_list:{output_filepath_list}")
        identifier_filepath_list = []
        for output_filepath in output_filepath_list:
            if self.identifier.toString() in output_filepath:
                identifier_filepath_list.append(output_filepath)

        # print(f"is_identical_code:self.identifier.toString():{self.identifier.toString()}")
        # print(f"is_identical_code:identifier_filepath_list:{identifier_filepath_list}")
        for identifier_filepath in identifier_filepath_list:
            with open(identifier_filepath, encoding="utf-8") as f:
                another_code = f.read()
            try:
                if is_same_syntax_by_ast(ast.parse(output_code), ast.parse(another_code)):
                    return True
            except SyntaxError:
                return True
            if self.input_mode_class.is_identical_syntax_by_cocode(output_code, another_code):
                return True
        return False

    def is_include_prioritization_line(self):
        if isinstance(self.mask_location, int):
            return int(self.identifier.start) <= self.mask_location <= int(self.identifier.end)
        elif isinstance(self.mask_location, list):
            for i in self.mask_location:
                if int(self.identifier.start) <= i <= int(self.identifier.end):
                    return True
            return False

    def iterate_all_files(self, target: str, model: str, prioritization:bool):
        """
        :param target: Target code path is for counting number of lines
        """
        target_without_extension = target.replace(".txt", "")
        input_file_path_list = None
        target_code_path = get_target_code_path(target_without_extension)
        if self.test_mode:
            test_input_file_path = f"{root}/test_input/{target_without_extension}/{model}/{target_without_extension}"
            self.perform_infilling_and_verification(target_code_path, test_input_file_path, prioritization)
        self.test_mode = False
        self.check_exception()
        if self.mode == "single":
            # input_file_path_list = list_txt_files(
            #     f"{root}/input/c1/simply_replace/{target_without_extension}") + list_txt_files(
            #     f"{root}/input/c2/add_specific/{target_without_extension}")
            input_file_path_list = list_txt_files(f"{root}/input/c1/simply_replace/{target_without_extension}")
        elif self.mode == "ast":
            input_file_path_list = list_txt_files(f"{root}/input/c1/ast/{target_without_extension}")
        is_correct_code = (target_without_extension[-1] == 'c')
        for each_file_path in input_file_path_list:
            self.identifier = self.mode_class.set_identifier(each_file_path)
            if self.dp_mode.is_target_identifier(self.identifier):
                if (isinstance(self.dp_mode, DP)
                        or (not self.mask_location and (not is_correct_code or (is_correct_code and self.identifier.depth != 's')))
                        or self.is_include_prioritization_line()):
                    start_time = time.time()
                    self.perform_infilling_for_one_file(each_file_path)
                    elapsed_time = time.time() - start_time
                    with open(f"{root}/time/{model}_{target}", "a", encoding="utf-8") as time_file:
                        time_file.write(f'{self.identifier.toString()}: {timedelta(seconds=elapsed_time)}\n')

    def perform_infilling_and_verification(self, target_code_path: str, path_to_code_to_be_infilled: str, prioritization_arg:bool):
        print("Prompting LLM...")
        with open(target_code_path, 'r', encoding="utf-8") as file2:
            original_code = file2.read()
        docstring = self.input_mode_class.read_docstring()
        docstring = docstring_convention(docstring)

        if self.test_mode:
            concatenated_content = (
                    "Generate diverse inputs for the target code. They should be different from example. Use '|' as delimiter to divide test inputs.\n\n## Docstring:\n"
                    + docstring + "## Target code\n```python\n" + original_code + "\n```"
            )
            self.generate_test_input(
                target_code_path, concatenated_content, f"{path_to_code_to_be_infilled}_div.txt")
            concatenated_content = (
                    "Generate complex inputs for the target code. They should be different from example. Use '|' as delimiter to divide test inputs.\n\n## Docstring:\n"
                    + docstring + "## Target code\n```python\n" + original_code + "\n```"
            )
            self.generate_test_input(
                target_code_path, concatenated_content, f"{path_to_code_to_be_infilled}_com.txt")
        else:
            with open(f'{path_to_code_to_be_infilled}', 'r', encoding="utf-8") as file2:
                masked_CUT = file2.read()
            concatenated_content = (
                    "Now, infill <MASK> of the following given code based on the following docstring.\n\n## Docstring:\n"
                    + docstring + "\n\n## Given code:\n```python\n" + masked_CUT + "\n```")
            self.generate_infilled_code_snippets(masked_CUT, target_code_path, path_to_code_to_be_infilled, 10,
                                                 concatenated_content, prioritization=prioritization_arg, pass_already_exist=False)

    def perform_infilling_for_one_file(self, each_file_path):
        # target_filename = os.path.basename(each_file_path)
        # print(f"Working on {target_filename}...")
        target_without_extension = self.target.replace(".txt", "")
        target_code_path = get_target_code_path(target_without_extension)
        generated_already = self.check_if_generated_before(each_file_path)
        # additional_filter = not "916c_9_add_specific" in each_file_path
        additional_filter = False
        if generated_already or additional_filter:
            print("!!!Hard-coded filter!!!")
            return
        print(f"each_file_path:{each_file_path}")
        self.mask_length = self.mode_class.get_mask_length(each_file_path)
        self.perform_infilling_and_verification(target_code_path, each_file_path, self.prioritization)

    def prompt_llm(self, concatenated_content, temperature=1.0):
        if self.model == "gpt":
            output_code = prompt_chatgpt(concatenated_content, temperature)
        elif self.model == "gemini":
            output_code = prompt_gemini(concatenated_content, temperature)
        elif self.model == "qwen":
            output_code = prompt_ali(concatenated_content, temperature)
        elif self.model == "alillama":
            output_code = prompt_ali_llama(concatenated_content, temperature)
        elif "deepseek" in self.model or "qwenv2" in self.model or "starchat" in self.model:
            output_code = self.prompt_hf_model(concatenated_content, temperature)
        elif "llama3-perplexity" in self.model:
            output_code = self.prompt_perplexity(concatenated_content, temperature)
        else:
            raise ValueError(f"Incorrect model choice {self.model}")
        return output_code

    def choose_one_shot_example(self, input_code_path):
        input_code_path = os.path.basename(input_code_path)

        #get line num from input
        focal_line_num = None
        if self.mode == "single":
            if "classeval" in input_code_path:
                # print(f"!!!target:{output_file_name}")
                focal_line_num = input_code_path.split("_")[3]
            elif "evo" in input_code_path:
                focal_line_num = input_code_path.split("_")[1]
            else:
                raise ValueError("Invalid line number")
        else:
            focal_line_num = input_code_path.split("_")[-3]

        focal_line = None
        with open(get_target_code_path(self.subject),'r') as f:
            for line_idx, each_line in enumerate(f.readlines(),1):
                if int(line_idx) == int(focal_line_num):
                    focal_line = each_line
        

        one_shot_example_file_name = None
        # print(f"focal_line{focal_line}")
        # print(f"focal_line_num{focal_line_num}")
        # print(f"mutation/subjects/target_code/{target}")
        if "if" in focal_line:
            if os.path.isfile(f"mutation/subjects/prompt/example_function_if_condition_full_{self.model}.txt"):
                one_shot_example_file_name = f"mutation/subjects/prompt/example_function_if_condition_full_{self.model}.txt"
            else:
                one_shot_example_file_name = "mutation/subjects/prompt/example_function_if_condition_full.txt"
        elif "for" in focal_line:
            if os.path.isfile(f"mutation/subjects/prompt/example_function_full_{self.model}.txt"):
                one_shot_example_file_name = f"mutation/subjects/prompt/example_function_full_{self.model}.txt"
            else:
                one_shot_example_file_name = f"mutation/subjects/prompt/example_function_full.txt"
        else:
            if os.path.isfile(f"mutation/subjects/prompt/example_function_full_{self.model}.txt"):
                one_shot_example_file_name = f"mutation/subjects/prompt/example_function_full_{self.model}.txt"
            else:
                one_shot_example_file_name = f"mutation/subjects/prompt/example_function_full.txt"

        return one_shot_example_file_name



        
        # template_path = ""
        # if template == None:
        #     template_path = f'{root}/prompt/example_{self.input_mode}.txt'
        # else:
        #     template_path = f'{root}/prompt/{template}.txt'        

    #The docstring and target code should be included in concatenated_content by the caller of this function
    def generate_infilled_code_snippets(
            self, masked_CUT, target_code_path, path_to_code_to_be_infilled: str, repeat, concatenated_content,
            temperature=1.0, prioritization=False, pass_already_exist=False):
        with open(target_code_path, encoding="utf-8") as f:
            CUT = f.read()
        # print("entering generate_infilled_code_snippets")
        one_shot_example_path = self.choose_one_shot_example(path_to_code_to_be_infilled)

        with open(one_shot_example_path, encoding="utf-8") as f:
            example_content = f.read()
        concatenated_content = f"{example_content}\n\n{concatenated_content}"
        if prioritization == True:
            self.end_no = 1
        generate_code_no = list(range(self.start_no, self.end_no + 1))
        print(f"!!!generate_code_no:{generate_code_no}!!!")
        for ith_code_generated in generate_code_no:
            if self.check_if_generated_before(path_to_code_to_be_infilled, ith_code_generated):
                # print(f"self.check_if_generated_before(path_to_code_to_be_infilled, ith_code_generated):{self.check_if_generated_before(path_to_code_to_be_infilled, ith_code_generated)}")
                print(f"Generated before, skipping...")
                continue
            self.prompting_llm_for_one_snippet(
                CUT, masked_CUT, path_to_code_to_be_infilled, ith_code_generated, concatenated_content,temperature,
                prioritization, pass_already_exist)

    # def ensure_code_is_merged_correctly():

    #Retrieve differing objects
    def extract_relevant_code_snippet_and_construct_complete_code(self,mask_start:int, mask_end, masked_CUT:str,
                                                                  extracted_code:str):
        # print(f"!!!mask_location:{mask_location}")
        # print(f"!!!masked_CUT:{masked_CUT}")
        # print(f"!!!extracted_code:{extracted_code}")
        diff_object = diff(mask_start, mask_end, masked_CUT.splitlines(),extracted_code.splitlines())
        if diff_object != 0:
            return "\n".join(diff_object.C2)
        else:
            return None

    def prompting_llm_for_one_snippet(
            self, CUT, masked_CUT, input_file_path, ith_code_generated, concatenated_content, temperature=1.0,
            prioritization=False, pass_already_exist=False):
        print(f"Generating {ith_code_generated}th code...")
        if not self.output_code_path or prioritization == True:
            output_code_path = self.creating_output_code_path(input_file_path, ith_code_generated, prioritization)
        else:
            output_code_path = self.output_code_path
        # print(f"!!!prioritization is :{prioritization}")
        # print(f"!!!output path from creating_output_code_path is :{output_code_path}")
        num_looping = 0
        generated_complete_code = False
        is_identical_code = False
        exception_retry_time = 0
        maximum_retry = 10
        while not generated_complete_code or is_identical_code:
            if num_looping == maximum_retry:
                if self.retry_time == 0 or self.retry_time == 1:
                    self.retry_time += 1
                    masked_CUT_temp = None
                    if self.mode == "ast":
                        masked_CUT_temp = "\n".join(ast_mutation(
                            CUT.splitlines(), input_file_path.split("_")[-3],
                            input_file_path.split("_")[-2], self.retry_time))
                    elif "simply_replace" in input_file_path:
                        masked_CUT_temp = "\n".join(ast_mutation(
                            CUT.splitlines(), input_file_path.split("_")[-4],
                            input_file_path.split("_")[-4], self.retry_time))
                    if self.retry_time == 1:
                        print("Using 1st template.")
                    else:
                        print("Using 2nd template.")

                    # This part is for retrying, the example template applies mask, the prompt without template is a few lines above.
                    with open(f"{root}/prompt/example_template.txt", encoding="utf-8") as f:
                        example_content = f.read()
                    docstring = self.input_mode_class.read_docstring()
                    concatenated_content_temp = (
                        example_content + "\n\n" +
                        "Now, infill <MASK> of the following given code based on the following docstring.\n\n" +
                        "## Docstring:\n" + docstring + "\n\n## Given code:\n```python\n" + masked_CUT_temp + "\n```")
                    self.prompting_llm_for_one_snippet(
                        CUT, masked_CUT_temp, input_file_path, ith_code_generated, concatenated_content_temp, prioritization=prioritization,
                        pass_already_exist=pass_already_exist)
                self.retry_time = self.syntax_retry_time
                return
            output_code = None
            # if True:
            # if pass_already_exist:
            #     if os.path.isfile(f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}"):
            if pass_already_exist and os.path.isfile(f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}"):
                    # print(f"!!!Infilling disabled for already existed raw output!!!")
                    output_code = open(f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}","r").read()
                # else:
                #     break
            else:
                # if prioritization == True and os.path.isfile(f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}"):
                #     output_code = open(f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}","r").read()
                # else:
                output_code = self.prompt_llm(concatenated_content, temperature)

            print(f"num_looping is {num_looping}...")
            if output_code == None:
                continue
            # print(f"==========num_looping is {num_looping}...==========")
            extracted_code = remove_empty_line_and_comment_from_code(extract_code(output_code))
            # print(f"extracted code:{extracted_code}")
            # print(f"output code is {output_code}, extracted code is {extracted_code}")
            assert extracted_code != None, f"extracted code is None, find out the reason, the output code path is {os.path.basename(output_code_path)}, output_code_is_{output_code}"
            pruned_infilled_extracted_code = None
            if not self.identifier:
                #for docstring)_verification
                # print(f"os.path.basename(output_code_path) is {os.path.basename(output_code_path)} and os.path.basename(output_code_path)[:3] is {os.path.basename(output_code_path)[:3]}")
                #filename starts with "dv_" indicates file for docstring verification
                if os.path.basename(output_code_path)[:3] == "dv_":
                    prefix_length = 4 + len(os.path.basename(output_code_path).split("_")[1])
                    self.identifier = Identifier(os.path.basename(output_code_path).split("_")[-2], get_line_num(os.path.basename(output_code_path)[prefix_length:],"ast")[0], get_line_num(os.path.basename(output_code_path)[prefix_length:],"ast")[1])
                    # print(f"self.identifier.start:{self.identifier.start}")
                    # print(f"self.identifier.end:{self.identifier.end}")
                else:
                    #otherwise, this branch handles prioritization
                    assert prioritization == True, "This assignment of self.identifier should only be applied to prioritization = True"
                    self.identifier = Identifier(os.path.basename(output_code_path).split("_")[-2], get_line_num(os.path.basename(output_code_path),"simply_replace")[0], get_line_num(os.path.basename(output_code_path),"simply_replace")[1])
            try:
                infilled_extracted_code = None
                #If the generated code is a full code, we extract only the infilled part, to avoid including the non-infilled part into the calculation of syntax diversity
                if is_full_code(extracted_code, CUT, self.mask_length):
                    # print("======print self attributes========")
                    # pprint(vars(self))
                    # if prioritization == True:
                    #     self.identifier = Identifier(None, get_line_num(os.path.basename(output_code_path)), get_line_num(os.path.basename(output_code_path)))
                        # self.identifier.start = self.identifier.end = get_line_num(output_file_name)
                    extracted_code_snippet_from_complete_code = self.extract_relevant_code_snippet_and_construct_complete_code(
                        int(self.identifier.start), int(self.identifier.end), masked_CUT, extracted_code)
                    if not extracted_code_snippet_from_complete_code and not prioritization:
                        print(f"extracted_code_snippet_from_complete_code:{extracted_code_snippet_from_complete_code}")
                        #===========For debugging===========
                        with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/masked_CUT.txt", "w") as f:
                            f.write(masked_CUT)
                        with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/extracted_code.txt", "w") as f:
                            f.write(extracted_code)
                        assert False, f"Understand why diff is None, identifier is {self.identifier.start}-{self.identifier.end}."
                        #===========For debugging===========
                        num_looping += 1
                        continue
                    infilled_extracted_code = snippet_infilling(masked_CUT, extracted_code_snippet_from_complete_code)
                pruned_extracted_code = refine_extracted_code_for_mask_location(masked_CUT.splitlines(),
                                                                            extracted_code.splitlines())
                pruned_infilled_extracted_code = snippet_infilling(masked_CUT, pruned_extracted_code)
                # print(f"after snippet infilling...")
            except Exception as e:
                traceback.print_exc()
                print(e)
                if not os.path.isdir("/data/toli/State-Level-DP/mutation/subjects/exception"):
                    os.makedirs("/data/toli/State-Level-DP/mutation/subjects/exception", exist_ok=True)
                with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/masked_CUT.txt", "w") as f:
                    f.write(masked_CUT)
                with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/extracted_code.txt", "w") as f:
                    f.write(extracted_code)
                exception_retry_time += 1
                if exception_retry_time == maximum_retry:
                    assert False, f"There is a bug in snippet_infilling, the output code path is {os.path.basename(output_code_path)}"
            exception_retry_time = 0
            # if prioritization:
            try:
                exec(pruned_infilled_extracted_code, None, {})
                infilled_extracted_code = pruned_infilled_extracted_code
            except Exception as e:
                print(f"There was an error while pruning code: {e}")
                num_looping += 1
                continue

            # if "evo1_22_simply_replace_1" in input_file_path:
            #     print(extracted_code)
            # print(f"*****path_to_code_to_be_infilled:{input_file_path}")
            # print(f"*****ith_code_generated:{ith_code_generated}")

            # Write prompt to file

            if not os.path.isdir(f"{os.path.dirname(output_code_path)}/prompt"):
                os.makedirs(f"{os.path.dirname(output_code_path)}/prompt", exist_ok=True)
            with open(f"{os.path.dirname(output_code_path)}/prompt/{os.path.basename(output_code_path)}", "w") as f:
                f.write(concatenated_content)
            # print(f"write to prompt...")
            # print(f"output_code_path:{output_code_path}")
            generated_complete_code = None
            # if prioritization == True:
            #Originally 
            if False:
                generated_complete_code = True
            else:
                generated_complete_code = self.verifying_complete_code(infilled_extracted_code)
                #===========For debugging===========
                # if not generated_complete_code:
                #     with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/infilled_extracted_code.txt", "w") as f:
                #         f.write(infilled_extracted_code)
                #     with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/extracted_code.txt", "w") as f:
                #         f.write(extracted_code)
                #     with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/masked_CUT.txt", "w") as f:
                #         f.write(masked_CUT)
                #     assert False, f"Understand why verifying_complete_code None, identifier is {self.identifier.start}-{self.identifier.end}."
                #===========For debugging end===========
            is_identical_code = False
            # print(f"!!!prioritization:{prioritization}")
            if prioritization == False:
                # print(f"!!!Verifying whether code is indentical")
                # if "7_7_4" in output_code_path:
                #     infilled_extractedcode = open("/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/evo15/starchat/1/dv_1_evo15_8_ast_7_7_4_1.txt","r").read()
                is_identical_code = self.is_identical_code(infilled_extracted_code,output_code_path)
            # assert not is_identical_code, f"is_identical_code:{is_identical_code}\ndebug:\n{infilled_extracted_code}\n{output_code_path}"
            # print(f"!!!is_identical_code:{is_identical_code}")
            if generated_complete_code and not is_identical_code and len(infilled_extracted_code) > 0:
                print(f"prioritization is:{prioritization}")
                print(f"the output path is:{output_code_path}")
                with open(output_code_path, "w", encoding="utf-8") as output_file:
                    print(f"writing to infilled code:{output_code_path}")
                    # if infilled_extracted_code:
                    output_file.write(infilled_extracted_code)
                    # else:
                    #     output_file.write(infilled_extracted_code)
                raw_output_path = f"{os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}"
                if not os.path.exists(raw_output_path):
                    with open(raw_output_path, "w", encoding="utf-8") as output_file:
                        output_file.write(output_code)
                # assert False, f"write to file: {os.path.dirname(output_code_path)}/raw_output/{os.path.basename(output_code_path)}"
                if prioritization == True:
                    break
            # Prevent infinite loop
            num_looping += 1

    def prompt_hf_model(self,concatenated_content: str, input_temperature:int=1.0):
        output = None
        if "starchat" in self.model:
            messages = [
                {
                    "role": "system",
                    "content": "You are StarChat2, an expert programming assistant",
                },
                {"role": "user", "content": concatenated_content},
            ]
            if input_temperature > 0.0:
                outputs = self.pipe(messages,max_new_tokens=1024,do_sample=True,temperature=input_temperature,top_k=50,top_p=0.95,stop_sequence="<|im_end|>")
            else:
                outputs = self.pipe(messages,max_new_tokens=1024,do_sample=False,top_k=50,stop_sequence="<|im_end|>")
            return outputs[0]["generated_text"][-1]["content"]
        elif "qwenv2" in self.model:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": concatenated_content}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

            if input_temperature > 0.0:
                generated_ids = self.model_loader.generate(model_inputs.input_ids,max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95, temperature=input_temperature)
            else:
                generated_ids = self.model_loader.generate(model_inputs.input_ids,max_new_tokens=1024, do_sample=False, top_k=50)

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif "deepseekv2" in self.model:
            messages = [
                {"role": "user", "content": concatenated_content}
            ]
            input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            outputs = self.model_loader.generate(input_tensor.to(self.model_loader.device), max_new_tokens=1024)

            result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            return result
        elif "deepseekr1" in self.model:
            messages = [
                {"role": "user", "content": concatenated_content}
            ]
            input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if torch.cuda.is_available():
                input_tensor = input_tensor.to("cuda")
            temperature = 0.6
            if input_temperature == 0.0:
                temperature = 0.0
            outputs = self.model_loader.generate(
                input_tensor.to(self.model_loader.device), max_new_tokens=2048, temperature=temperature)
            result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            return result
        elif "llama3-8b" in self.model:
            # print(f"!!!Prompt Llama 3.")
            dialogs: List[Dialog] = [
                [{"role": "user", "content": concatenated_content}]
            ]
            results = self.pipe.chat_completion(
                dialogs,
                max_gen_len=1024,
                temperature=input_temperature,
                top_p=0.9
            )
            for dialog, result in zip(dialogs, results):
                return result['generation']['content']
        else:
            messages = [
                {"role": "user", "content": concatenated_content}
            ]
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model_loader.device)
            # tokenizer.eos_token_id is the id of <|EOT|> token
            #temperature has to be positive i.e., > 0.0, otherwise need to set do_sample False
            if input_temperature > 0.0:
                outputs = self.model_loader.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95, temperature=input_temperature, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
            else:
                #activating do_sample means unsetting top_p
                outputs = self.model_loader.generate(inputs, max_new_tokens=1024, do_sample=False, top_k=50, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    def prompt_perplexity(self, concatenated_content: str, input_temperature:int=1.0):
        YOUR_API_KEY = "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"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    concatenated_content
                ),
            },
        ]

        client = OpenAI(api_key=mutation.api.llama_api_key, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
        model="llama-3.1-8b-instruct",
        temperature=input_temperature,
        messages=messages,
        )
        return response.choices[0].message.content

    @func_set_timeout(120)
    def verify_test_case(self, output_code, test_cases: str):
        test_case_list = test_cases.split("|")
        if isinstance(self.input_mode_class, GenerateFuncWithLLM):
            for idx in range(len(test_case_list) - 1, -1, -1):
                if not self.input_mode_class.verify(output_code, test_case_list[idx]):
                    test_case_list.pop(idx)
            if test_case_list:
                return test_case_list
            else:
                return False
        elif isinstance(self.input_mode_class, GenerateClassWithLLM):
            if test_cases == "":
                print("Empty input")
                return False
            output, err = utils.class_execute(output_code, test_case_list)
            return not err
    
    def verifying_complete_code(self, output_code: str) -> bool:
        print("Verifying code...")
        output_code_lines = output_code.splitlines()
        if not output_code_lines:
            print(f"Debug: {output_code}")
        # generated_complete_code = abs(len(original_code.splitlines()) - len(output_code.splitlines())) <= 5 #Better not use string matching, because string can be vastly different
        if "__main__" in output_code:
            return False
        
        test_input = self.input_mode_class.get_test_input()

        generated_complete_code = self.input_mode_class.verify(output_code, test_input)
        if generated_complete_code:
            print("Successfully generated complete code...")
        else:
            print("Fail to generate complete code, need generation again...")
            # with open("/data/toli/State-Level-DP/mutation/subjects/exception/unverified_code.py", "w", encoding="utf-8") as f:
            #     f.write(output_code)
        return generated_complete_code


class GenerateCodeWithLLama3_8b(GenerateCodeWithLLM):
    def __init__(self, target: str, model: str, mode: str, test_mode, input_mode, dp_mode, prioritization=False, mask_location=None):
        super().__init__(target, model, mode, test_mode, input_mode, dp_mode, prioritization, mask_location)

    def initialise_hf_model(self):
        print(f"!!!Reminder: remember to set global variable 'Rank', 'WORLD_SIZE', \
                    'MASTER_ADDR' and 'MASTER_PORT', and also run prioritization.py by \
                    !!!torchrun --nproc_per_node 1 mutation/prioritization.py!!!!!!Note that this command is to run mutation/prioritization.py!!!")
        # export RANK=1
        # export WORLD_SIZE=1 #replication
        # export MASTER_ADDR=127.0.0.1
        # export MASTER_PORT=29500
        self.pipe = Llama.build(
            ckpt_dir="/data/toli/llama3/Meta-Llama-3-8B-Instruct/",
            tokenizer_path="/data/toli/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model",
            max_seq_len=2048,
            max_batch_size=1,
        )
        print(f"Finish loading model")

    def prompt_llm(self, concatenated_content):
        output_code = self.prompt_hf_model(concatenated_content)
        return output_code


class GenerateCodeBySingle:
    def set_identifier(self, each_file_path):
        each_filename_with_extension: str = each_file_path.split("/")[-1]
        mask_location = each_filename_with_extension.split("_")[1]
        return Identifier(None, mask_location, mask_location)

    def get_mask_length(self, each_file_path=None):
        return 1


class GenerateCodeByAst:
    def set_identifier(self, each_file_path):
        each_file_path_split = each_file_path.split("/")[-1].removesuffix(".txt").split("_")[-3:]
        return Identifier(each_file_path_split[2], each_file_path_split[0], each_file_path_split[1])

    def get_mask_length(self, each_file_path):
        starting_line = each_file_path.split("_")[-3]
        end_line = each_file_path.split("_")[-2]
        return int(end_line) - int(starting_line) + 1


class GenerateConsoleWithLLM:
    def __init__(self, target, model):
        self.model = model
        self.target = target
        self.subject = target.removesuffix(".txt")

    def get_test_input(self):
        return get_test_input(self.subject, self.model)

    def verify(self, output_code, test_input):
        output_code_lines = output_code.splitlines()
        output_code_lines = wrap_as_function(output_code_lines, analyze_indentation(output_code), "verify")
        with open("stub_folder/verify_generated_code.py", "w", encoding="utf-8") as file:
            file.write("\n".join(output_code_lines))
        try:
            subp = subprocess.Popen(["python", "stub_folder/verify_generated_code.py"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = subp.communicate(input=test_input.encode(), timeout=10)
            if subp.poll() is None:
                subp.terminate()
                raise TimeoutError
            if err != b'':
                raise SyntaxError(err.decode())
        except TimeoutError:
            print("TimeoutError")
            return False
        except SyntaxError:
            return False
        return True

    def read_docstring(self):
        return read_docstring(self.target)


class GenerateFuncWithLLM:
    def __init__(self, target, func_name, model):
        self.func_name = func_name
        self.model = model
        self.target = target
        self.subject = target.removesuffix(".txt")

    def read_docstring(self):
        return read_docstring(self.target)

    def get_test_input(self):
        if self.model == "llama3-8b":
            print("!!!USING starchat's input!!!")
            self.model = "starchat"
        return get_test_input(self.subject, self.model)
    
    def is_identical_syntax_by_cocode(self, code1, code2: str):
        globals_dict = {}
        func_name2 = f"{self.func_name}_e"
        code2_r = code2.replace(self.func_name, func_name2)
        exec(code1, globals_dict)
        exec(code2_r, globals_dict)
        func1 = globals_dict[self.func_name]
        func2 = globals_dict[func_name2]
        return func1.__code__.co_code == func2.__code__.co_code

    def verify(self, output_code, test_input):
        if test_input == "":
            print("Empty input")
            return False
        globals_dict = {}
        output_code_lines = output_code.splitlines()
        try:
            func_def = 0
            for index, line in enumerate(output_code_lines):
                if self.func_name in line and "def" in line:
                    func_def = index
            output_timeout_lines = (
                    output_code_lines[:func_def] + ["from func_timeout import func_set_timeout",
                                                    "@func_set_timeout(30)"] +
                    output_code_lines[func_def:])  # add timeout between import and function def
            output_with_timeout = "\n".join(output_timeout_lines)
            exec(output_with_timeout, globals_dict)
            for each_input in test_input.splitlines():
                my_function = globals_dict[self.func_name]
                if each_input != "":
                    eval_args = eval(each_input)
                    if isinstance(eval_args, tuple) and (each_input[0] != "(" or each_input[-1] != ")"):
                        my_function(*eval_args)
                    else:
                        my_function(eval_args)
        except SyntaxError as e:
            traceback.print_exc()
            return False
        except FunctionTimedOut:
            print("Timeout Error")
            return False
        except Exception as e:
            traceback.print_exc()
            return False
        return True


class GenerateClassWithLLM:
    def __init__(self, gcllm):
        gcllm.test_mode = False
        self.subject: str = gcllm.subject.removesuffix("_" + gcllm.subject.split("_")[-1])

    def update_subject(self, subject:str):
        self.subject = subject

    def read_docstring(self):
        with open(f"{root}/prompt/{self.subject}.txt", encoding="utf-8") as f:
            return f.read()

    def get_test_input(self):
        with open(f"{root}/playground_template/{self.subject}.txt", encoding="utf-8") as file:
            infilled_code = file.read()
        return infilled_code
    
    def is_identical_syntax_by_cocode(self, code1, code2: str):
        # globals_dict = {}
        # func_name_list = get_class_function_name(code1)
        # class_name1 = get_class_name(code1)
        # class_name2 = f"{class_name1}E"
        # code2_r = code2.replace(class_name1, class_name2)
        # exec(code1, globals_dict)
        # exec(code2_r, globals_dict)
        # for func_name in func_name_list:
        #     func1 = globals_dict[f"{class_name1}.{func_name}"]
        #     func2 = globals_dict[f"{class_name2}.{func_name}"]
        #     if func1.__code__.co_code != func2.__code__.co_code:
        #         return False
        # return True
        return False

    @staticmethod
    def verify(output_code, infilled_code):
        '''
        This function execute code using test cases by obtain_output()
        '''
        if "<generated_code_here>" not in infilled_code:
            print("<generated_code_here> not found. Infilling would be to no avail!")
            raise KeyError
        infilled_code = infilled_code.replace("<generated_code_here>", output_code)
        class_name = get_class_name(output_code)
        try:
            exec(infilled_code, globals())
        except FunctionTimedOut:
            print("Timeout Error")
            globals().pop(class_name)
            return False
        except Exception as e:
            traceback.print_exc()
            globals().pop(class_name)
            return False
        globals().pop(class_name)
        return True


class DP:
    def __init__(self, identifier):
        self.identifier = identifier

    def get_start_end(self):
        return 11, 20

    def is_target_identifier(self, identifier):
        if identifier.toString() == self.identifier:
            return True
        else:
            return False


class Normal:
    def get_start_end(self):
        return 1, 10

    def is_target_identifier(self, identifier):
        return True


def constructGCLLM(target: str, model: str, mode: str, test_mode, input_mode="function", dp_mode=Normal(), prioritization=False, mask_location=None):
    if "llama3-8b" in model:
        return GenerateCodeWithLLama3_8b(target, model, mode, test_mode, input_mode, dp_mode, prioritization, mask_location)
    else:
        return GenerateCodeWithLLM(target, model, mode, test_mode, input_mode, dp_mode, prioritization, mask_location)


def main(target: str, model: str, mode, dp_mode, func_name=None, mask_location=None):
    initialisation(target, model)
    file_check(target)
    if dp_mode:
        dp_mode_class = DP(dp_mode)
    else:
        dp_mode_class = Normal()
    subject = target.removesuffix(".txt")
    args = load_args(subject)
    input_mode = args["subject_type"]
    if not mask_location:
        mask_location = args["mask_location"]
    gcllm = constructGCLLM(target, model, mode, True, input_mode, dp_mode_class, mask_location=mask_location)
    gcllm.initialise_hf_model()
    if mode == "single":
        gcllm.mode_class = GenerateCodeBySingle()
    else:
        gcllm.mode_class = GenerateCodeByAst()
    if input_mode == "console":
        gcllm.input_mode_class = GenerateConsoleWithLLM(target, model)
    elif input_mode == "function":
        gcllm.input_mode_class = GenerateFuncWithLLM(target, func_name, model)
    else:
        gcllm.input_mode_class = GenerateClassWithLLM(gcllm)
    gcllm.iterate_all_files(target, model, False)


if __name__ == '__main__':
    # target = "886d.txt"
    target = sys.argv[1]
    model = sys.argv[2]  # "gpt" or "gemini"
    mode = sys.argv[3]  # single or ast
    input_mode = sys.argv[4]
    main(target, model, mode, input_mode, False)
