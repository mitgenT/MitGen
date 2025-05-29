import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CrossVerificationDemo import class_execute, FunctionExecute
import perform_mutation as pm
import generate_code_with_llm as gcllm
import socket
from mutation.generate_code_with_llm import GenerateCodeWithLLM, GenerateCodeWithLLama3_8b, docstring_convention, GenerateConsoleWithLLM, GenerateFuncWithLLM,list_txt_files_without_subdir,GenerateCodeBySingle, GenerateClassWithLLM, Normal
import importlib
import difflib
from Levenshtein import distance
from ast_mutator import AstMutator
from statistics import mean
from io import StringIO
import subprocess
from mutation.utils import is_equal, list_txt_files, load_args, diff, get_line_num, \
    remove_empty_line_and_comment_from_code, get_target_code_path, create_identifiers, coarse_grain
import json
import ast
import csv
'''
This program performs mutation + code generation. Please run the individual programs for individual functions.
'''
sys.set_int_max_str_digits(0)


def get_mask_block(target_id_without_extension,target_line):
    identifiers = create_identifiers(coarse_grain(target_id_without_extension, "ast"))
    involved_block = []
    for identifier in identifiers:
        if identifier.start <= target_line <= identifier.end:
            involved_block.append([identifier.start, identifier.end, identifier.end - identifier.start])
    return involved_block


def assess_revelant_blocks_performance (revelant_block_list):
    result_dict:dict[str,int] = {"Failure revealed!":0,"false positive":0}
    for each_result_line in revelant_block_list:
        # print(f"each_result_line:{each_result_line}")
        # print(f"each_result_line.split:{each_result_line.4split('[[')}")
        # print(f"each_result_line:{each_result_line.split("[[")}")
        #Handle the case that no code generated for an input-output pair
        if len(each_result_line.split("[[")) == 1:
            continue
        else:
            result_type = each_result_line.split("[[")[1].split(",")[0]
            if "not verified" in result_type:
                continue
            if result_type in result_dict:
                result_dict[result_type] += 1
    return result_dict

def get_egrv_performance_by_involved_block(target_id_without_extension,model,line_index,using_generated_failing_inpt_arg2=False):
    egrv_all_lines = None
    if using_generated_failing_inpt_arg2:
        assert os.path.exists(f"/data/toli/State-Level-DP/output/{target_id_without_extension}{model}_sd_g.txt"), f"Reference version result for {target_id_without_extension} with model {model} has not been generated."
        egrv_all_lines = open(f"/data/toli/State-Level-DP/output/{target_id_without_extension}{model}_sd_g.txt","r").readlines()
    else:
        assert os.path.exists(f"/data/toli/State-Level-DP/output/{target_id_without_extension}{model}_sd.txt"), f"Reference version result for {target_id_without_extension} with model {model} has not been generated."
        egrv_all_lines = open(f"/data/toli/State-Level-DP/output/{target_id_without_extension}{model}_sd.txt","r").readlines()        

    block_start = True
    block_end = False
    revelant_block_list: list[str] = []
    for each_line in egrv_all_lines:
        #not input_outputs + match line index -> initiate collecting egrv performance
        if "Input-Output Pair" not in each_line:
            if line_index in each_line:
                block_start = True
            else:
                block_start = False
            continue
        if  block_start == True:
            revelant_block_list.append(each_line)
    a_block_result_dict = assess_revelant_blocks_performance(revelant_block_list)
    return a_block_result_dict



def evaluate_prioritization_quality(target_id_without_extension,model,using_generated_failing_inpt_arg1=False):
    prioritization_result = open(f'mutation/subjects/output/c1/prioritization/{target_id_without_extension}/{model}/dataframe/final.csv', 'r').read().splitlines()
    # print(f"prioritization_result:{prioritization_result}")
    #each ranked_each_line is a tuple (line,syntax similarity score)
    # print(f"prioritization_result:{prioritization_result}")
    overall_result_dict = None
    prioritization_result_to_analyze = prioritization_result[:5]
    # if buggy_line != -1:
    #     prioritization_result_to_analyze = [f"{buggy_line},0"]
    overall_result_dict = {}
    for idx, ranked_each_line in enumerate(prioritization_result_to_analyze,1):
        
        
        found_inconsistency_egrv = False
        #avoid empty lines
        if len(ranked_each_line) > 0:
            # print(f"ranked_each_line:{ranked_each_line}")
            prioritized_line = int(ranked_each_line.split(",")[0])
            involved_block = get_mask_block(target_id_without_extension,prioritized_line)
            # involved_block = [(int(ranked_each_line.split(",")[0]),int(ranked_each_line.split(",")[0]))] #single line
            # print(f"involved_block:{involved_block}")
            involved_block.sort(key=lambda x: x[2],reverse = True)
            
            for each_involved_block in involved_block:
                
                # print(f"ranked_each_line:{prioritized_line},involved_block:{involved_block},each_involved_block:{each_involved_block}")
                start_line = each_involved_block[0]
                end_line = each_involved_block[1]
                line_index = f"{start_line}_{end_line}"
                
                # get performance
                tp_fp_dict = get_egrv_performance_by_involved_block(target_id_without_extension,model,line_index,using_generated_failing_inpt_arg2 = using_generated_failing_inpt_arg1)
                overall_result_dict[f"{line_index}({idx},{prioritized_line})"] = tp_fp_dict
                # if tp_fp_dict['Failure revealed!'] != 0 or tp_fp_dict['false positive'] != 0:
                #     found_inconsistency_egrv = True
            #No inconsistency-inducing test is found, try another prioritized line
            # if found_inconsistency_egrv:
            #     break
    print(overall_result_dict)
    return overall_result_dict

    # get_mask_block(target_id_without_extension,target_line):
    # print(ranked_each_line)#returns the key
    # get_mask_block(target_code,target_line)
    # with open(f'mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe/final.csv', 'w') as csv_file:  
    #     writer = csv.writer(csv_file)
    #     for key, value in remove_same_syntax_and_same_output_dict.items():
    #         writer.writerow([key, value])

    # mask_blocks = []
    # # CUT_lines = CUT.splitlines()
    # tree = ast.parse(CUT)
    # for child in ast.iter_child_nodes(tree):
    #     # if isinstance(child, ast.ClassDef):
    #     mask_blocks.append([child.lineno + 1, child.end_lineno])
    # print(f"mask_blocks:{mask_blocks}")
    # for mask_block in mask_blocks:
    #     masked_code = "\n".join(ast_mutation(CUT_lines, mask_block[0], mask_block[1]))
    #     with open(
    #             f"{root}/input/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{mask_block[0]}_{mask_block[1]}_0.txt",
    #             "w", encoding="utf-8") as f:
    #         f.write(masked_code)


# def select_egrv_based_single_line(target_line):

#     ast_node_line_ranges = 
    # line

def find_line_of_first_def(masked_CUT):
    function_signature_line_idx = -1
    masked_CUT_list = masked_CUT.splitlines()
    for idx, each_line in enumerate(masked_CUT_list):
        if "def" in each_line:
            function_signature_line_idx = idx
    assert function_signature_line_idx != -1, "function_signature_line_idx has not found"
    return function_signature_line_idx

def constructing_concatenated_content(target,masked_CUT,function_signature_line_idx,docstring):
    #To handle the case that classeval's target files have multiple individual bug (but each bug correspond the the same docstring)


    #constructing concatenated_content
    # if "evo" in target:
    #     '''
    #     Note:
    #     ''.join(masked_CUT.splitlines(keepends=True)[:function_signature_line_idx + 1]) is function signature
    #     ''.join(masked_CUT.splitlines(keepends=True)[function_signature_line_idx + 1:]) is function body
    #     '''
    #     concatenated_content = (
    #                     "Now, infill <MASK> of the following given code.\n\n"
    #                     + "## Given code:\n```python\n" + ''.join(masked_CUT.splitlines(keepends=True)[:function_signature_line_idx + 1]) + docstring + "\n" + ''.join(masked_CUT.splitlines(keepends=True)[function_signature_line_idx + 1:]) + "\n```")
    # if "classeval" in target:
    concatenated_content = (
        "Now, infill <MASK> of the following given code.\n\n"
        + f"##Docstring:\n{docstring}\n\n## Given code:\n\n```python\n{masked_CUT}\n```")
    return concatenated_content

def construct_paths_to_code_to_be_infilled(target,mode):
    #To replace special characters
    target_without_extension = target.replace(".txt", "")
    input_file_path_list = list_txt_files(f"mutation/subjects/input/c1/{mode}/{target_without_extension}")
    return input_file_path_list

def obtain_docstring(target):
    if "classeval" in target:
        filename_of_docstring = '_'.join(target.split("_")[:2]) + ".txt"
        print(f"filename_of_docstring:{filename_of_docstring}")
    else:
        filename_of_docstring = target
    with open(f'mutation/subjects/prompt/{filename_of_docstring}', 'r', encoding="utf-8") as file1:
        docstring = file1.read()
    docstring = docstring_convention(docstring)
    return docstring

def prioritization_infilling(target:str,model:str,mode:str,input_mode,gcllm: GenerateCodeWithLLM,pass_already_exist):
    print("Prompting LLM...")
    if pass_already_exist == True:
        print("!!!Disable infilling for already existed raw output!!!")
    filename_of_docstring = None
    gcllm.update_target(target)
    gcllm.mode_class = GenerateCodeBySingle()
    # gcllm.iterate_all_files(target, model)
    #load model to GPU
    docstring = obtain_docstring(target)
    input_file_path_list = construct_paths_to_code_to_be_infilled(target,"simply_replace")        
    #`each_file_path` is a full path
    for each_file_path in input_file_path_list:
        print(f"Perform infilling for {os.path.basename(each_file_path)}...")
        gcllm.mask_length = gcllm.mode_class.get_mask_length(each_file_path)
        # print(f"each_file_path:{each_file_path}")
        with open(each_file_path, 'r', encoding="utf-8") as file2:
            masked_CUT = file2.read()
        #Find the line of function signature
        function_signature_line_idx = find_line_of_first_def(masked_CUT)
        #Constructing prompt
        actual_task_prompt_instruction = constructing_concatenated_content(target,masked_CUT,function_signature_line_idx,docstring)
        retry_time = 1
        # print(f"each_file_path:{each_file_path}")
        #path to be infilled only need prefix up to subject directory, e.g., mutation/subjects/input/c1/simply_replace/evo2, no needd to the rest
        # print("Try joining input")
        # print('/'.join(each_file_path.split("/")[:-2]))
        assert retry_time >= 1, "Retry time includes first trial, hence should be >= 1."
        gcllm.generate_infilled_code_snippets(
            masked_CUT, get_target_code_path(target.removesuffix(".txt")), each_file_path, retry_time,
                                              actual_task_prompt_instruction, 0.0, True, pass_already_exist)

def print_list(target_list):
    for idx, each_element in enumerate(target_list):
        print(f"{idx},{each_element}")

def print_list(target_dict):
    for idx, each_key in enumerate(target_dict):
        print(f"{idx+1},{each_key},{target_dict[each_key]}")


def diff_lines(cut:list[str],generated:list[str],target_line:str):
    """
    This function output differing lines between cut and generated code.

    :param target_line: line that we are interested, to filter lines that we are not interested in
    :returns: this is a description of what is returned
    :raises keyError: raises an exception
    """
    print(f"******target line is:{target_line}")
    # for idx_cut, each_cut_line in enumerate(cut):
    #     for idx_generated, each_generated_line in enumerate(generated):
    d = difflib.Differ()
    diffs_generator = d.compare(cut,generated)
    # print_list(diff)
    # diffs_generator = difflib.unified_diff(cut, generated)
    # diffs == [] means there is no difference between two compared code
    diffs = [i for i in diffs_generator]
    udpated_line = []

    #Handle the case that there is no difference
    if diffs == []:
        return []
    
    #target_line_idx is to mark the idx of target line, in order to perform forward or backward comparison
    target_line_idx = None
    #diffs is a generator, so needs to convert to list first
    print_list(diffs)
    
    # print(f"len of diff:{len(diffs)}")
    for idx, each_line in enumerate(diffs):

        #We may not need to consider whether the line is legit, the logic in the code can detect it
        is_legit_line = True
        # is_legit_line = not ("@@" in diffs[idx] or diffs[idx][0] == '?') and len(diffs[idx]) > 0
        # is_legit_line = not ("@@" in diffs[idx] or "+++" in diffs[idx] or "---" in diffs[idx]) and len(diffs[idx]) > 0
        if is_legit_line:
            #raw line means line without annotation or leading spaces
            raw_line = ""

            #Find the idx of target line, split into two cases, the first case is that target line has been removed
            is_target_line_removed = False
            if each_line[0] == '-':
                raw_line = each_line[1:].strip()
                is_target_line_removed = True
            else:
                raw_line = each_line.strip()
            
            print(f"raw_line:{raw_line}")
            print(f"target_line_idx:{target_line_idx}")
            print(f"raw_line in target_line:{raw_line in target_line}")
            if raw_line in target_line:
                target_line_idx = idx
            
            # raw_line may equal '---', which does not exist in the original code and cause target_line_idx to be None
            if target_line_idx != None:
                # print(f"******Investigating legit line, idx:{idx},each_line:{each_line}")
                # print(f"******raw_line:{raw_line}")
                # print(f"******target idx is:{target_line_idx}")
                
                #perform backward line searching
                # is_legit_line = target_line_idx - 1 >= 0 and not ("@@" in diffs[target_line_idx - 1] or "+++" in diffs[target_line_idx - 1] or "---" in diffs[target_line_idx - 1])

                #we do not need to consider the case of minus, becaus minus can only be done in original program
                while target_line_idx - 1 >= 0 and (diffs[target_line_idx - 1][0] == "+" or diffs[target_line_idx - 1][0] == "?"):
                    if diffs[target_line_idx - 1][0] == "?":
                        target_line_idx -= 1
                        continue
                    udpated_line.append(diffs[target_line_idx - 1][1:])
                    target_line_idx -= 1

                    # is_legit_line = target_line_idx - 1 >= 0 and not ("@@" in diffs[target_line_idx - 1] or "+++" in diffs[target_line_idx - 1] or "---" in diffs[target_line_idx - 1])
                
                target_line_idx = idx
                # print(f"target_line_idx + 1:{target_line_idx + 1}")
                # print(f"len(diffs):{len(diffs)}")
                
                # print(f"******target_line_idx + 1:{target_line_idx + 1}")
                # print(f"******len(diffs) - 1:{len(diffs) - 1}")
                # print(f"******target_line_idx:{target_line_idx}")
                # print(f"******is_legit_line:{is_legit_line}")
                # print(f"******foward search while loop condition:{is_legit_line and (diffs[target_line_idx + 1][0] == "+" or diffs[target_line_idx + 1][0] == "?" or diffs[target_line_idx + 1][0] == "-")}")
                # print(f"diffs[target_line_idx + 1]:{diffs[target_line_idx + 1]}")
                # is_legit_line = target_line_idx + 1 <= len(diffs) - 1 and not ("@@" in diffs[target_line_idx + 1] or "+++" in diffs[target_line_idx + 1] or "---" in diffs[target_line_idx + 1])
                # test = is_legit_line and (diffs[target_line_idx + 1][0] == "+" or diffs[target_line_idx + 1][0] == "-")

                #we do not need to consider the case of minus, becaus minus can only be done in original program
                # is_non_code_line = 
                is_legit_line = target_line_idx + 1 <= len(diffs) - 1
                print(f"target_line_idx + 1:{target_line_idx + 1}")
                print(f"len(diffs):{len(diffs)}")
                print(f"is_legit_line:{is_legit_line}")
                # print(f"diffs[target_line_idx + 1][0]:{diffs[target_line_idx + 1][0]}")
                loop_count_debug = 0
                while is_legit_line and (diffs[target_line_idx + 1][0] == "+" or diffs[target_line_idx + 1][0] == "?" or diffs[target_line_idx + 1][0] == "-"):
                    print(f"DEBUG: in loop {loop_count_debug}")
                    loop_count_debug += 1
                    if diffs[target_line_idx + 1][0] == "?" or diffs[target_line_idx + 1][0] == "-":
                        target_line_idx += 1
                        continue
                    udpated_line.append(diffs[target_line_idx + 1][1:])
                    target_line_idx += 1
                    is_legit_line = target_line_idx + 1 <= len(diffs) - 1
                    # print(f"is_legit_line:{is_legit_line and (diffs[target_line_idx + 1][0] == "+" or diffs[target_line_idx + 1][0] == "?" or diffs[target_line_idx + 1][0] == "-")}")
                    
                    
                    # is_legit_line = target_line_idx + 1 <= len(diffs) - 1 and not ("@@" in diffs[target_line_idx + 1] or "+++" in diffs[target_line_idx + 1] or "---" in diffs[target_line_idx + 1])
                # print(f"line_update:{line_update}")
                break
    assert target_line_idx != None, "target_line_idx equals None, there are problem with the matching algorithm"
    
    #handle the case that the only change is target line gets removed
    if is_target_line_removed and udpated_line == []:
        udpated_line.append("")
    return udpated_line    
        #todo: computing the diff between target line and line_update
        # print("Done...")

#diff_cut_line is a dictionary stores the edit distance of each generated code compared to buggy code
def eval_prioritization_result(target_subject:str,diff_cut_line,line_num):
    # remove_0_dict = {x:y for x,y in diff_cut_line.items() if y!=0}
    # sorted_dict = {k: v for k, v in sorted(remove_0_dict.items(), key=lambda item: item[1], reverse = True)}
    sorted_dict = {k: v for k, v in sorted(diff_cut_line.items(), key=lambda item: item[1])}
    gt_from_file = open(f"mutation/subjects/prioritization_gf/{target_subject}",'r').readlines()
    gt = [int(i) for i in gt_from_file]
    rank_of_target_line = []
    for each_gt in gt:
        for rank,each_top_n in enumerate(sorted_dict): #this returns the key 
            # print(f"each_gt,each_top_n:{each_gt},{each_top_n}")
            if each_gt == each_top_n:
                rank_of_target_line.append(rank + 1)
    total = len(diff_cut_line)
    if rank_of_target_line == []:
        return (line_num,f"rank/total:(filtered out)/{total}")
        # print(f"rank/total:(filtered out)/{total}")
    else:
        final_rank = mean(rank_of_target_line)
        return (line_num,f"rank/total:{final_rank}/{total}")
        
    

def get_top_n(diff_cut_line,rank_n):
    remove_0_dict = {x:y for x,y in diff_cut_line.items() if y!=0}
    # sorted_dict = {k: v for k, v in sorted(remove_0_dict.items(), key=lambda item: item[1], reverse = True)}
    sorted_dict = {k: v for k, v in sorted(remove_0_dict.items(), key=lambda item: item[1])}
    print(f"sorted_dict:{sorted_dict}")
    top_n = []
    for i, k in enumerate(sorted_dict):
        # if i < rank_n:
            top_n.append(k)
    return top_n


def computing_distance(target_line:str, generated_code:str):
    #There are modification on target line
    if len(target_line) > 0:
        return distance(target_line, generated_code)/len(target_line)
    else:
        return distance(target_line, generated_code) #There are addition only
    # cumulative_diff = 0
    # for each_line in generated_line_list:
    #     cumulative_diff += distance(target_line, each_line)
    # return cumulative_diff

def get_line_and_compare_syntax(target_subject:str,model:str):

    #diff_cut_line stores the difference between each cut line and its generated line
    diff_cut_line = {}

    target_without_extension = target_subject.replace(".txt", "")
    prioritization_output_file = f"mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}"
    cut_path = get_target_code_path(target_without_extension)
    # cut_code = open(cut_path,"r").readlines()
    cut_code_full = open(cut_path,"r").read()
    if not os.path.exists(prioritization_output_file):
        os.makedirs(prioritization_output_file, exist_ok=True)
    prioritization_file_path_list = list_txt_files_without_subdir(prioritization_output_file)
    #`each_file_path` is a full path
    # prioritization_file_path_list = ['/Users/jy/Documents/chatgpt/State-Level-DP/mutation/subjects/output/c1/prioritization/evo2/gpt/evo2_2_simply_replace_1_1.txt']


    for iteration_idx,each_generated_code_path in enumerate(prioritization_file_path_list):
        if "/raw_output/" in each_generated_code_path or "/prompt/" in each_generated_code_path:
            continue
        # print(f"==============START==============")
        # print(f"******Handling file:{each_generated_code_path}")
        output_file_name = os.path.basename(each_generated_code_path)
        # generated_code = open(each_generated_code_path,"r").readlines()
        generated_code_full = open(each_generated_code_path,"r").read()
        target_line_idx,_ = get_line_num(output_file_name,"simply_replace")
        # print(f"******target_line_idx:{target_line_idx}")

        #Don't add remove_empty_line_and_comment_from_code for diff_value's argument, because the mask location after refined will be in a different location from the specified one.
        try:
            diff_value = diff(int(target_line_idx),int(target_line_idx), cut_code_full.splitlines(), generated_code_full.splitlines())
        except Exception as e:
            print(e)
            if not os.path.isdir("/data/toli/State-Level-DP/mutation/subjects/exception"):
                os.makedirs("/data/toli/State-Level-DP/mutation/subjects/exception", exist_ok=True)
            with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/prioritization_cut_code_full.txt", "w") as f:
                f.write(cut_code_full)
            with open(f"/data/toli/State-Level-DP/mutation/subjects/exception/prioritization_generated_code_full.txt", "w") as f:
                f.write(generated_code_full)
        # if int(target_line_idx) == 31:
        #     print("====================")
        #     print(f"int(target_line_idx):{int(target_line_idx)}")
        #     print(f"iteration_idx:{iteration_idx}")
        #     print(f"cut_path:{cut_path}")
        #     print(f"each_generated_code_path:{each_generated_code_path}")
        #     print(f"remove_empty_line_and_comment_from_code(cut_code_full):{remove_empty_line_and_comment_from_code(cut_code_full).splitlines()}")
        #     print(f"remove_empty_line_and_comment_from_code(generated_code_full):{remove_empty_line_and_comment_from_code(generated_code_full).splitlines()}")
        #     print(f"diff_value = diff(int(target_line_idx), remove_empty_line_and_comment_from_code(cut_code_full).splitlines(), remove_empty_line_and_comment_from_code(generated_code_full).splitlines()):{diff(int(target_line_idx), remove_empty_line_and_comment_from_code(cut_code_full).splitlines(), remove_empty_line_and_comment_from_code(generated_code_full).splitlines())}")
        #     print("====================")
        # if "classeval_assessmentsystem_24_44" in each_generated_code_path:
        #     print(f"!!!diff_value:{diff_value}")
    
        if diff_value == 0:
            diff_cut_line[target_line_idx] = 0.0
        else:
            # if int(target_line_idx) == 31:
            # print(f"diff_value.locations:{diff_value.locations}")
            # print(f"diff_value.C1:{diff_value.C1}")
            # print(f"diff_value.C1:{diff_value.C2}")

            diff_cut_line[target_line_idx] = computing_distance(remove_empty_line_and_comment_from_code("\n".join(diff_value.C1)),remove_empty_line_and_comment_from_code("\n".join(diff_value.C2)))

            # if "classeval_assessmentsystem_24_44" in each_generated_code_path:
            #     print(f"!!!diff_cut_line[target_line_idx]:{diff_cut_line[target_line_idx]}")
            # print(f"C1 code:{"\n".join(diff_value.C1)}")
            # print(f"C2 code:{"\n".join(diff_value.C2)}")
            # print(f"distance:{diff_cut_line[target_line_idx]}")
            # print("====================")
    sorted_dict = {k: v for k, v in sorted(diff_cut_line.items(), key=lambda item: item[1])}
    return sorted_dict
        # print(f"******cut_code:{cut_code}")
        # generated_line_list = diff_lines(cut_code,generated_code,cut_code[target_line_idx-1])
        # print(f"******generated_line_list:{generated_line_list}")
        # # 
        # if generated_line_list == []:
        #     diff_cut_line[target_line_idx-1] = 0.0
        # else:
        #     diff_cut_line[target_line_idx-1] = computing_distance(cut_code[target_line_idx-1],generated_line_list)
            # break
    # top_n = get_top_n(diff_cut_line,1)
    # return eval_prioritization_result(target_subject,cut_code_full,diff_cut_line,line_num)
        
        # break
        # print(file_name)
        # output_file = (each_file_path.replace("/input/", "/output/")
        #             .replace(f"/{target_subject}/", f"/{target_subject}/{model}/")
        #             .replace(".txt", f"_{gen_index}.txt"))
        # output_path = 


# def eval():
    #For each infilled line, check whether line's syntax (edit distance)
    #filter out that has 0 distance, and pick the one (AST block) with the smallest distance
    #see whether the picked AST block includes the buggy line

# def main_manual():
#     print("Input target file, e.g., 886d.txt")
#     # python mutation/prioritization.py evo2.txt gpt single 
#     # python mutation/prioritization.py evo2.txt deepseek67 single 
#     # python mutation/prioritization.py classeval_assessmentsystem_1.txt deepseek13 single 
#     target = sys.argv[1] #e.g., 886d.txt
#     model = sys.argv[2] #e.g., gpt or gemini
#     mode = sys.argv[3] #e.g., single or ast
#     # input_mode = sys.argv[4] #e.g., console or function
#     if "classeval" in target:
#         input_mode = "console"
#     elif "evoeval" in target:
#         input_mode = "function"
#     else:
#         raise ValueError("Unhandled input mode")
#     # Step 1: Constructing prompt
#     pm.main(target,model,mode, input_mode)
#     # # Step 2: Infilling
#     prioritization_infilling(target,model,mode, input_mode)
#     # Step 3: Get lines and output rank (for all mask location)
#     get_line_and_compare_syntax(target,model)

def derive_input_mode(target_dataset):
    input_mode = None
    if "classeval" in target_dataset:
        input_mode = "class"
    elif "evoeval" in target_dataset:
        input_mode = "function"
    else:
        raise ValueError("Unhandled input mode")
    return input_mode

def construct_gllm(target, model,input_mode,skip_already_exist):
    gcllm = GenerateCodeWithLLM(target, model, "single", True, input_mode, Normal(),prioritization=True)
    # if not skip_already_exist: 
    if "deepseek" in model or "qwenv2" in model or "starchat" in model:
        # print(f"initialising tokenizer")
        gcllm.initialise_hf_model()
    elif "llama3-8b" in model:
        gcllm = GenerateCodeWithLLama3_8b(target, model, "single", True, input_mode, Normal(),prioritization=True)
        gcllm.initialise_hf_model()
    return gcllm

def return_function_name():
    dict = {
        "evo1":"separate_paren_groups",
        "evo1c":"separate_paren_groups",
        "evo2":"truncate_number",
        "evo2c":"truncate_number",
        "evo5":"intersperse",
        "evo5c":"intersperse",
        "evo6":"parse_nested_parens",
        "evo6c":"parse_nested_parens",
        "evo10":"make_palindrome",
        "evo10c":"make_palindrome",
        "evo11":"string_xor_advanced",
        "evo11c":"string_xor_advanced",
        "evo12":"longest_substring",
        "evo12c":"longest_substring",
        "evo13":"multiple_greatest_common_divisors",
        "evo13c":"multiple_greatest_common_divisors",
        "evo14":"all_prefix_suffix_pairs",
        "evo14c":"all_prefix_suffix_pairs",
        "evo15":"string_sequence_modified",
        "evo15c":"string_sequence_modified",
        "evo18":"how_many_times",
        "evo18c":"how_many_times",
        "evo26":"remove_duplicates_and_count",
        "evo26c":"remove_duplicates_and_count",
        "evo27":"flip_case_special",
        "evo27c":"flip_case_special",
        "evo28":"interleave_and_concatenate",
        "evo28c":"interleave_and_concatenate",
        "evo29":"filter_by_prefix_suffix",
        "evo29c":"filter_by_prefix_suffix",
        "evo34":"unique",
        "evo34c":"unique",
        "evo36":"advanced_fizz_buzz",
        "evo36c":"advanced_fizz_buzz",
        "evo37":"sort_even_odd",
        "evo37c":"sort_even_odd",
        "evo39":"prime_fib_matrix",
        "evo39c":"prime_fib_matrix",
        "evo41":"car_race_collision",
        "evo41c":"car_race_collision",
        "evo43":"multi_pairs_sum_to_zero",
        "evo43c":"multi_pairs_sum_to_zero",
        "evo44":"change_base",
        "evo44c":"change_base",
        "evo46":"fib4_memo",
        "evo46c":"fib4_memo",
        "evo49":"modp",
        "evo49c":"modp",
        "evo48":"is_palindrome_sentence",
        "evo48c":"is_palindrome_sentence",
        "evo51":"remove_vowels_and_count",
        "evo51c":"remove_vowels_and_count",
        "evo55":"fib",
        "evo55c":"fib",
        "evo58":"common",
        "evo58c":"common",
        "evo64":"vowels_count",
        "evo64c":"vowels_count",
        "evo65":"circular_shift",
        "evo65c":"circular_shift",
        "evo67":"advanced_fruit_distribution",
        "evo67c":"advanced_fruit_distribution",
        "evo70":"strange_sort_list",
        "evo70c":"strange_sort_list",
        "evo72":"will_it_fly_advanced",
        "evo72c":"will_it_fly_advanced",
        "evo73":"smallest_change",
        "evo73c":"smallest_change",
        "evo75":"is_multiply_prime",
        "evo75c":"is_multiply_prime",
        "evo76":"complex_power",
        "evo76c":"complex_power",
        "evo77":"iscube",
        "evo77c":"iscube",
        "evo78":"hex_key_primes",
        "evo78c":"hex_key_primes",
        "evo81":"numerical_letter_grade_with_weightage",
        "evo81c":"numerical_letter_grade_with_weightage",
        "evo82":"prime_length",
        "evo82c":"prime_length",
        "evo84":"enhanced_solve",
        "evo84c":"enhanced_solve",
        "evo86":"advanced_anti_shuffle",
        "evo86c":"advanced_anti_shuffle",
        "evo88":"sort_array",
        "evo88c":"sort_array",
        "evo91":"detect_boredom",
        "evo91c":"detect_boredom",
        "evo96":"prime_sequences",
        "evo96c":"prime_sequences"
    }
    return dict

def construct_subject_name_including_txt(target_dataset,each_target_id):
    target = None
    if "evoeval" in target_dataset:
        target = f"evo{each_target_id}.txt"
    elif "classeval" in target_dataset:
        target = f"{target_dataset}_{each_target_id}.txt"
    else:
        raise ValueError("Invalid dataset name.")
    return target

def main_auto(dataset_list:list[str], subject_dict:dict[str,list[int]],need_generation:bool,model):
    print("!!!Running automatic mode!!!") 
    '''
    models:
    - deepseek67
    - deepseek13
    - deepseek33
    - qwen
    - starchat
    - llama3-8b
    - llama3-70b
    - codellama-33b
    '''
    input_mode = None #input_model can be equal to None except for llama3-8b
    skip_already_exist = False
    func_name_dict = return_function_name()
    #evo
    # evo_remove_list = [44,75]
    # target_id_list = [5,44,46,73]
    # target_id_list = [46]
    # target_id_list = [10]
    #time out for buggy code, no need to test: 44
    #crash for buggy code, no need to test: 75
    # target_id_list = [46,55,67,73,78,81,84]
    # target_id_list = [6,11,12,13,14,15,18,26,27,28,37,39,46,55,67,73,78,81,84]
    # target_id_list = [1]
    # target_id_list = [84]
    # target_id_list = [x for x in target_id_list if x not in evo_remove_list]
    
    #classEval
    # target_id_list = [37,39,44,46,55,67,73,75,78,81,84]
    # target_id_list = subject_list
    # target_id_list = [5,6,7,8,9,10,11,12,13,14,15,18,24,25,28,35,36,37,38]
    for dataset in dataset_list:
        target_dataset = None
        if "evo" in dataset:
            target_dataset = "evoeval"
        elif "classeval" in dataset:
            target_dataset = "classeval"
        # target_dataset = "classeval"
        # model = 'deepseek33'
        # model = 'qwen'
        # model = 'llama3-8b'
        mode = 'single'
        input_mode = derive_input_mode(target_dataset)
        print(f"!!!target_dataset is {target_dataset}, input_mode is {input_mode}")
        for each_target_id in subject_dict[dataset]:

            #e.g., target = "evo16.txt"
            target = construct_subject_name_including_txt(dataset,each_target_id)
            if need_generation:
                gcllm = construct_gllm(target, model, input_mode, skip_already_exist)
            if input_mode == "class" and need_generation:
                gcllm.input_mode_class = GenerateClassWithLLM(gcllm)
            elif input_mode == "function" and need_generation:
                gcllm.input_mode_class = GenerateFuncWithLLM(target,func_name_dict[target.replace(".txt","")],model)
            gcllm.input_mode = input_mode
            # line_num = get_line_num(target)
            # target = f"classeval_assessmentsystem_{each_target_id}.txt"
            print(f"=====Working on target{each_target_id}=====")
            # Step 1: Constructing prompt
            if need_generation:
                # skip_already_exist = False
                if input_mode == "class":
                    pm.main(target, model, None, mode, input_mode)
                else:
                    pm.main(target,model,func_name_dict[target.replace(".txt","")], mode, input_mode)
                print(f"Finish mutation...")
                # assert False, "debugging"
                # # Step 2: Check whether test input has been generated
                subject = target.replace(".txt", "")
                if target_dataset == "evoeval" and len(list_txt_files_without_subdir(f"/data/toli/State-Level-DP/mutation/subjects/test_input/{subject}/{model}")) < 20:
                    get_test_input_starchat(gcllm, subject, model, input_mode)
                # # Step 3: Infilling
                prioritization_infilling(target,model,mode, input_mode,gcllm,skip_already_exist)
                print(f"Finish generating code...")
            # # Step 3: Get lines and output rank
            line_to_syntax_distance_dict = get_line_and_compare_syntax(target,model) 

            #Compute out distance, very wired that the following code works while calling obtain_generated_code_output cannot
            line_to_output_distance_dict = {}
            target_without_extension = target.replace(".txt","")

            #Note that list_txt_files is a recursive approach, so the output may contain raw_output or backup
            all_prioritization_output_file = list_txt_files_without_subdir(f"mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}")
            # print(f"all_prioritization_output_file:{all_prioritization_output_file}")
            #get buggy code output
            buggy_code_path = get_target_code_path(target_without_extension)
            # print(f"!!!buggy_code_output:{execute_code_and_obtain_output(target,buggy_code_path)}")

            # Execute buggy code and obtain output of buggy code
            buggy_output_string = execute_code_and_obtain_output(target,buggy_code_path)
            assert buggy_output_string != None, f"Buggy output is None, there is some problem with mutation/executable/{target.replace('.txt','.py')} (e.g., the original bug causes crash, hang), or haven't added :::::: to print output (Need to add 'Final output of subject for prioritization is:::::' to print output)"
            # print(f"buggy_output_string:{buggy_output_string}")
            
            #Handle the case of crash/exception
            # buggy_code_output = None
            
            # if buggy_code_output != None:
            buggy_code_output = eval(buggy_output_string)
            buggy_code_output_count = len(buggy_code_output)

            #compute generated output distance w.r.t. output code
            for each_output_file in all_prioritization_output_file:
                if "raw_output" in each_output_file or "backup" in each_output_file or "prompt" in each_output_file:
                    continue
                line_num, output_distance = get_output_distance_of_generated_code(target,each_output_file,buggy_code_output)
                line_to_output_distance_dict[line_num] = output_distance
            # print(line_to_output_distance_dict)

            # line_to_output_distance_dict = obtain_generated_code_output(target,model,buggy_code_output)

            #Combine the two dict together
            line_to_all_distance_dict = {}
            for line_num in line_to_syntax_distance_dict:
                line_to_all_distance_dict[line_num] = (line_to_syntax_distance_dict[line_num],line_to_output_distance_dict[line_num])
            remove_uncompilable_dict = {x:y for x,y in line_to_all_distance_dict.items() if y[1]!=-1}
            print("========remove_uncompilable_dict========")
            print_list(remove_uncompilable_dict)
            remove_same_syntax_dict = {x:y for x,y in remove_uncompilable_dict.items() if y[0]>0.0}
            print("========remove_same_syntax_dict========")
            print_list(remove_same_syntax_dict)
            remove_same_syntax_and_same_output_dict = {x:y for x,y in remove_same_syntax_dict.items() if y[1]<buggy_code_output_count}
            print("========remove_same_syntax_and_same_output_dict========")
            print_list(remove_same_syntax_and_same_output_dict)
            os.makedirs(f"mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe",exist_ok=True)
            print(f"!!!Path to save csv:mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe")
            with open(f'mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe/final.csv', 'w') as csv_file:  
                writer = csv.writer(csv_file)
                for key, value in remove_same_syntax_and_same_output_dict.items():
                    writer.writerow([key, value])
            # print(remove_same_syntax_and_same_output_dict)
            # # print({k: v for k, v in sorted(line_to_all_distance_dict.items(), key=lambda item: item[0])})
            # print(line_to_all_distance_dict)
            # print("!!!Buggy code output is hard-coded!!!")

def ablation_study_of_componenet(line_to_all_distance_dict,remove_uncompilable_dict,remove_same_syntax_dict,remove_same_syntax_and_same_output_dict):
    print(f"All generated code:{len(line_to_all_distance_dict)}:100.00")
    print(f"Compilable code:{len(remove_uncompilable_dict)}:{len(remove_uncompilable_dict)/len(line_to_all_distance_dict)}")
    print(f"Compilarbale code and different syntax:{len(remove_same_syntax_dict)}:{len(remove_same_syntax_dict)/len(line_to_all_distance_dict)}")
    print(f"remove_same_syntax_and_same_output_dict:{len(remove_same_syntax_and_same_output_dict)}:{len(remove_same_syntax_and_same_output_dict)/len(line_to_all_distance_dict)}")


# def context_similarity():

    
    
def output_similarity_with_buggy(generated_code_output,buggy_code_output):
    matched_output_count = 0
    if len(generated_code_output) != len(buggy_code_output):
        return 0, True
    for k in range(len(buggy_code_output)):
        # print(f"generated_code_output:{generated_code_output[k]},buggy_code_output:{buggy_code_output[k]},is_equal:{is_equal(generated_code_output[k],buggy_code_output[k])}")
        # print(f"generated_code_output:{len(generated_code_output[k])},buggy_code_output:{len(buggy_code_output[k])}")
        if isinstance(generated_code_output[k][1],list) and isinstance(buggy_code_output[k][1],list):
            if len(generated_code_output[k][1]) == len(buggy_code_output[k][1]) and is_equal(generated_code_output[k][1],buggy_code_output[k][1])[0]:
                matched_output_count += 1
        else:
            # if isinstance(generated_code_output[k][1], type(buggy_code_output[k][1])) or isinstance(buggy_code_output[k][1], type(generated_code_output[k][1])):
            if is_equal(generated_code_output[k][1],buggy_code_output[k][1])[0]:
                matched_output_count += 1
    return matched_output_count, False


def execute_code_and_obtain_output(target,generated_code_path,line_num:str=-1) -> str:
    executable_code = []
    # print(f"!!!target:{generated_code_path}")
    # template_path = None
    # if "classeval" in generated_code_path:
    #     template_path = "mutation/classeval_playground_template.py"
    # elif "evo" in generated_code_path:
    #     template_path = 123
    # else:
    #     raise ValueError("Undefined template path.")
    target_template_name = None
    if "evo" in target:
        target_template_name = target
        target_template_name = target_template_name.replace("c.txt", ".txt")  # The replacement is to allow buggy code and correct code share the same template
        template_path = f"mutation/subjects/playground_template/{target_template_name}"
        if os.path.isfile(template_path):
            with open(template_path,"r") as f:
                lines = f.read()
                if "<generated_code_here>" in lines:
                    executable_code = lines.replace("<generated_code_here>", remove_empty_line_and_comment_from_code(open(generated_code_path,'r').read()))
                else:
                    raise Exception("Mask label not found")

            generated_code_basename = os.path.basename(generated_code_path).replace(".txt",".py")
            result = None
            os.makedirs("mutation/executable", exist_ok=True)
            with open(f"mutation/executable/{generated_code_basename}",'w') as f:
                f.write(''.join(executable_code))
            try:
                print(f"python mutation/executable/{generated_code_basename}")
                p = subprocess.run(f"python mutation/executable/{generated_code_basename}", shell=True, stdout=subprocess.PIPE, timeout=60)
                all_output = p.stdout.decode()
                # print(f"!!!Show all output:{all_output}")
                #Retrieve the real output from all printed messages in result.
                for each_line in all_output.splitlines():
                    if "Final output of subject for prioritization is" in each_line:
                        result = each_line.split(":::::")[1]
            except Exception as e:
                # dummy = 1
                print(e.output)
                result = e.output
                # print("!!!Entering exception")
            # print(f"mutation/executable/{generated_code_basename}")
            # print(f"!!!result:{result}")
            return result
        else:
            with open(f"mutation/subjects/arguments/{target}", encoding="utf-8") as f:
                arguments = f.read()
            result = []
            data = load_args(target.removesuffix(".txt"))
            execute_class = FunctionExecute(data["func_name"])
            for index, argument in enumerate(arguments.splitlines()):
                result.append((index, execute_class.execute(generated_code_path.removesuffix(".txt"), argument)))
            return str(result)
    elif "classeval" in target:
        module = generated_code_path.removesuffix(".txt")
        subject = target.removesuffix(".txt")
        return str(class_execute(module, subject))


def get_output_distance_of_generated_code(target,generated_code_path,buggy_code_output):
    output_file_name = os.path.basename(generated_code_path)
    line_num,_ = get_line_num(output_file_name,"simply_replace")
    result = execute_code_and_obtain_output(target,generated_code_path,line_num)
    if not result:
        # print("Exception")
        return line_num, -1
    else:
        # eval_result = result
        # print(f"result:{result}")
        # print(f"=========End=========")
        # print(f"output_file_name:{output_file_name}")
        eval_result = eval(result)
        similarity, err = output_similarity_with_buggy(eval_result,buggy_code_output)
        if err:
            return line_num, -1
        else:
            return line_num, similarity
        # eval(f"eval_result = {result}")
        # if eval_result == None:
        #     print(f"!!!output_file_name:{output_file_name} is None")
        #     print(f"!!!result is:{result}")
    # f = open("executable/classeval_playground_executable", "w")
    # f.write('\n'.join(executable_code))
    # f.close()

#output_target should be the filename that incldues line number

def get_test_input_starchat(gcllm: GenerateCodeWithLLM, target, model, input_mode):
    # if self.test_mode:
    func_sign_dict = return_function_name()
    # gcllm = GenerateCodeWithLLM(None, model, "single", True, input_mode, Normal())
    # gcllm.input_mode_class = GenerateFuncWithLLM(target,func_sign_dict[target],model)
    # gcllm.initialise_hf_model()
    # gcllm.update_target(target)
    # target = "evo27"
    target_code_path = get_target_code_path(target)
    original_code = open(target_code_path,"r").read()
    docstring = f"/data/toli/State-Level-DP/mutation/subjects/prompt/{target}.txt"
    output_path = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}/{target}_div.txt"
    concatenated_content = (
            "Generate diverse inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
            + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    gcllm.generate_test_input(target_code_path, concatenated_content, output_path)
    concatenated_content = (
                    "Generate complex inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
                    + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    output_path = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}/{target}_com.txt"
    gcllm.generate_test_input(target_code_path, concatenated_content, output_path)


if __name__ == "__main__":
    # s = "[(0, {'name': 'Alice', 'grade': 3, 'major': 'Mathematics', 'courses': {}}), (1, {'Alice': {'name': 'Alice', 'grade': 3, 'major': 'Mathematics', 'courses': {}}, 'Bob': {'name': 'Bob', 'grade': 2, 'major': 'Science', 'courses': {}}}), (2, {'Alice': {'name': 'Alice', 'grade': 3, 'major': 'Mathematics', 'courses': {}}, 'Bob': {'name': 'Bob', 'grade': 2, 'major': 'Science', 'courses': {}}, 'Charlie': {'name': 'Charlie', 'grade': 4, 'major': 'Chemistry', 'courses': {}}}), (3, {'Alice': {'name': 'Alice', 'grade': 3, 'major': 'Mathematics', 'courses': {}}, 'Bob': {'name': 'Bob', 'grade': 2, 'major': 'Science', 'courses': {}}, 'Charlie': {'name': 'Charlie', 'grade': 4, 'major': 'Chemistry', 'courses': {}}, 'David': {'name': 'David', 'grade': 1, 'major': 'Physics', 'courses': {}}}), (4, {'Alice': {'name': 'Alice', 'grade': 3, 'major': 'Mathematics', 'courses': {}}, 'Bob': {'name': 'Bob', 'grade': 2, 'major': 'Science', 'courses': {}}, 'Charlie': {'name': 'Charlie', 'grade': 4, 'major': 'Chemistry', 'courses': {}}, 'David': {'name': 'David', 'grade': 1, 'major': 'Physics', 'courses': {}}, 'Eve': {'name': 'Eve', 'grade': 3, 'major': 'Mathematics', 'courses': {}}}), (5, 90), (6, 90), (7, 95), (8, 85), (9, {}), (10, 85.0), (11, 'Exception'), (12, 'Exception'), (13, 'Exception'), (14, 90.0), (15, ['Bob']), (16, []), (17, []), (18, []), (19, ['Alice', 'Bob']), (20, 85.0), (21, 85.0), (22, None), (23, None), (24, 90.0), (25, 'Alice'), (26, 'Exception'), (27, None), (28, 'Bob'), (29, 'Bob'), (30, 'Exception'), (31, ['student 2']), (32, 84.0)]"
    # eval(s)
    # 46 takes too long

    # dataset = "evo" OR dataset = "classeval_assessmentsystem_"
    # dataset_list = ["classeval_sqlgenerator"]
    # dataset_list = ["classeval_accessgatewayfilter","classeval_excelprocessor"] #classeval_sqlgenerator
    model = sys.argv[1]
    dataset_list = ["evoeval"]
    # subject_dict = {"classeval_sqlgenerator": [32]}
    # subject_dict = {
    #     "classeval_accessgatewayfilter": ['6','7','11'],
    #     "classeval_excelprocessor":['9','11','34','39'],
    #     "classeval_sqlgenerator":['6','9','32'],
    #     # "evoeval":['5']
    # }
    subject_dict = {
        # "classeval_sqlgenerator": ['9']
        # "evoeval":['1','11','12','14','15','18','91']
        "evoeval":['11c']
        # "evoeval":['15c']
    }
    #sccpu6: not generated code,'18c'
    #complete sccpu6 list: '1c','11c','15c','37c','67c','82c','5c','48c','58c','65c','73c','88c','96c'
    #completed '5','11','12','13','14','15','18','27','28'
    #No input '6','37'
    #remaining '39','44','55','67','75','78','84'
    #ClassEval finished: ['3']
    # "evoeval":['1','2','5','6','11','12','13','14','15','18','26','27','28','37','39','44','46','55','67','73','75','78','81','84']
    # "evoeval":['15c','18c','37c','44c','48c']
    # subject_list = [7] 
    # subject_list = [34,36,41,49,70,77,82]
    # subject_list = [48,49,51,55,58,64,65,67]
    need_generation = True
    main_auto(dataset_list,subject_dict,need_generation,model)

    for each_dataset in dataset_list:
        for each_subject_id in subject_dict[each_dataset]:
            target_without_extension = None
            if "evoeval" in each_dataset:
                target_without_extension = f"evo{each_subject_id}"
            elif "classeval" in each_dataset:
                target_without_extension = f"{each_dataset}_{each_subject_id}"
            print(f"=============={target_without_extension}==================")
            overall_result_dict = evaluate_prioritization_quality(target_without_extension,model)
            with open(f"mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe/final_output.txt", "w", encoding="utf-8") as f:
                f.write(str(overall_result_dict))
            #Example of overall_result_dict: {'8_45(1,27)': {'Failure revealed!': 0, 'false positive': 0}, '27_27(1,27)': {'Failure revealed!': 0, 'false positive': 4}, '8_45(2,37)': {'Failure revealed!': 0, 'false positive': 0}, '36_40(2,37)': {'Failure revealed!': 5, 'false positive': 1}, '37_37(2,37)': {'Failure revealed!': 5, 'false positive': 0}
            # for each_key in overall_result_dict:
            #     involved_block = 
            #     get_egrv_performance_by_involved_block(target_without_extension,model,[2,17])

    # get_mask_block("evo73",9)

        # get_line_and_compare_syntax(target,model)
        # break
    # main_wo_input_evo()
    # main_manual()
    # main_auto()


#Test the import module
# filename = "evo2_4_simply_replace_1.t_verifymknyl1qz"
# func_name = "truncate_number"
# each_input = "3.5,2"
# module = importlib.import_module(f"stub_folder.{filename}")
# eval(f"module.{func_name}({each_input})")

