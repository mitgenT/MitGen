
import sys
import os
here = os.path.dirname(__file__)
import shutil
sys.path.append(os.path.join(here, '..'))
from mutation.generate_code_with_llm import extract_code,refine_extracted_code_for_mask_location,snippet_infilling,is_full_code,GenerateCodeWithLLM,GenerateFuncWithLLM,Normal
import CrossVerificationDemo 
from mutation.utils import diff,list_txt_files
from semantic_analysis import Semantic_Analysis
# import mutation.subjects.output.c1.ast.evo43.starchat.evo43_9_ast_6_8_3_1.multi_pairs_sum_to_zero
# import mutation.subjects.output.c1.ast.evo43.starchat.evo43_9_ast_6_8_3_1

# print(hhh.multi_pairs_sum_to_zero([1, 3, 5, 0], 2))
# print(mutation.subjects.output.c1.ast.evo43.starchat.evo43_9_ast_6_8_3_1.multi_pairs_sum_to_zero([1, 3, 5, 0], 2))


def construct_subject_functionSignature_dict():
    dict = {
        "evo18":"string_sequence_modified",
        "evo27":"flip_case_special",
        "evo28":"interleave_and_concatenate",
        "evo36":"advanced_fizz_buzz",
        "evo37":"unique",
        "evo39":"unique",
        "evo41":"car_race_collision",
        "evo43":"multi_pairs_sum_to_zero",
        "evo44":"change_base",
        "evo48":"is_palindrome_sentence",
        "evo55":"fib",
        "evo58":"common",
        "evo65":"circular_shift",
        "evo67":"advanced_fruit_distribution",
        "evo70":"strange_sort_list",
        "evo72":"will_it_fly_advanced",
        "evo75":"is_multiply_prime",
        "evo76":"complex_power",
        "evo77":"iscube",
        "evo78":"complex_hex_key_primespower",
        "evo82":"prime_length",
        "evo84":"enhanced_solve",
        "evo86":"advanced_anti_shuffle",
        "evo91":"detect_boredom"
    }
    return dict

def test_get_input_starchat(target):
    # if self.test_mode:
    func_sign_dict = construct_subject_functionSignature_dict()
    model = "starchat"
    input_mode = "function"
    gcllm = GenerateCodeWithLLM(None, model, "single", True, input_mode, Normal())
    gcllm.input_mode_class = GenerateFuncWithLLM(target,func_sign_dict[target],model)
    gcllm.initialise_hf_model()
    gcllm.update_target(target)
    # target = "evo27"
    target_code_path = f"/data/toli/State-Level-DP/mutation/subjects/target_code/{target}.txt"
    original_code = open(target_code_path,"r").read()
    docstring = f"/data/toli/State-Level-DP/mutation/subjects/prompt/{target}.txt"
    path_to_code_to_be_infilled = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/starchat/{target}_div.txt"
    concatenated_content = (
            "Generate diverse inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
            + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    gcllm.generate_test_input(target_code_path, concatenated_content, path_to_code_to_be_infilled)
    concatenated_content = (
                    "Generate complex inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
                    + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    path_to_code_to_be_infilled = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/starchat/{target}_div.txt"
    gcllm.generate_test_input(target_code_path, concatenated_content, path_to_code_to_be_infilled)


def txt_to_py(path:str):
    if not os.path.exists(path):
        assert os.path.exists(path.replace(".py",".txt"))
        shutil.copyfile(path.replace(".py",".txt"), path)

def test_mutated_input(target:str):
    model = "starchat"
    input_mode = "function" 
    target_signature_dict = construct_subject_functionSignature_dict()
    test_input_file_list = list_txt_files(f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}")
    sa_o = Semantic_Analysis("debug", target, input_mode, model, "linux")
    for idx, each_test_input in enumerate(test_input_file_list,1):
        # if  "test_api" in each_test_input:
        print(f"Working on idx:{idx}")
        input_from_file = open(each_test_input, encoding="utf-8").read()
        sa_o.mutate_input(input_from_file)
        #!!!Reminder: execute_tests only accepts relative path (e.g., mutation/subjects/correct_code/evo12.py)

        code1_path = f"mutation/subjects/correct_code/{target}.py"
        code2_path = f"mutation/subjects/target_code/{target}.py"
        txt_to_py(code1_path)
        txt_to_py(code2_path)

        result = sa_o.execute_tests([code1_path,code2_path],target, target_signature_dict[target],is_diff_testing=True)
        print(f"Test execution result:{result}")
            # print(f"each_test_input:{each_test_input}")

#subject=None,subject_type=None,model=None,verification_type=None,function_name=None,argument_list=None
def end_to_end_input_generation():
    arguments_list = ['147, 247', '1000, 2000', '150, 250', '123456, 654321', '999999, 1000000', '2, 200000', '250000, 500000', '100, 10000', '987654, 1000000', '1, 99999', '100000, 200000', '1, 10000', '200, 500', '500, 1000', '5000, 10000', '7000, 8000', '2500, 7500', '300, 700', '8000, 12000', '9000, 15000', '100, 400', '123456, 654321', '987654, 1234567', '111111, 666666', '222222, 777777', '333333, 888888', '444444, 999999', '10000, 20000', '50000, 60000', '100000, 200000', '500, 1000', '1000, 2000', '100, 500', '5000, 10000', '100, 20000']
    target_signature_dict = construct_subject_functionSignature_dict()
    subject="evo84"
    subject_type="function"
    model="starchat"
    verification_type="sd"
    function_name="enhanced_solve"
    CrossVerificationDemo.main(subject_arg=subject,subject_type_arg=subject_type,model_arg=model,verification_type_arg=verification_type,function_name_arg=function_name,argument_list_arg=arguments_list)



target = "evo91"
# ===========Step 1: Generate input with LLM===========
test_get_input_starchat(target)
# =======Step 2: Mutate input and find failing test===
# test_mutated_input(target)
# =======Evaluate end-to-end result==========
# end_to_end_input_generation()


# def test_diff():
#     mask_location = 31
#     target_cpde_lines = open("mutation/subjects/target_code/classeval_assessmentsystem_8.txt","r").readlines()
#     str2_lines = open("mutation/subjects/output/c1/prioritization/classeval_assessmentsystem_8/qwen/classeval_assessmentsystem_8_31_simply_replace_1_1.txt","r").readlines()
#     diff(test_diff,str1_lines,str2_lines)

# def test_input_generation():
#     subject = "evo27"
#     model = "starchat"
#     input_mode = "function"
#     sa = Semantic_Analysis("ast", subject, input_mode, model, "linux")
#     test_cases = sa.generate_test_input()
#     print(test_cases)


# test_input_generation()

'''
Below test infilling adaption.
'''


# subject_num = sys.argv[1]
# input_file_path_list = list_txt_files(f"/data/toli/SDP_prioritization/State-Level-DP/mutation/subjects/input/c1/simply_replace/evo{subject_num}")
# for each_input in input_file_path_list:
#     base_filename = os.path.basename(each_input)
#     line_num = base_filename.split("_")[1]
#     # subject_prefix = f"evo{subject_num}_"
#     # print(f"each_input:{each_input}")
#     # print(f"os.path.basename(each_input):{base_filename}")
#     # print(f"each_input:{subject_num}")
#     for each_output in list_txt_files(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/ast/evo{subject_num}/starchat"):
#         # print(f"subject_prefix:{subject_prefix}")
#         if f"_ast_{line_num}" in each_output:
#             print(f"each_output:{each_output}")
#             print(f"_ast_{line_num}")
#             masked_code_lines = open(each_input,"r").read()
#             extracted_code = extract_code(open(each_output,"r").read())
#             print(f"refined code:\n{snippet_infilling(masked_code_lines, extracted_code)}")



#=======ClassEval=======
# subject_num = sys.argv[1]
# line_num = sys.argv[2].
# masked_code_lines = open(f"mutation/subjects/input/c1/simply_replace/classeval_assessmentsystem_{subject_num}/classeval_assessmentsystem_{subject_num}_{line_num}_simply_replace_1.txt","r").read()
# extracted_code = extract_code(open(f"mutation/subjects/output/c1/prioritization/classeval_assessmentsystem_{subject_num}/deepseekv2/raw_output/classeval_assessmentsystem_{subject_num}_{line_num}_simply_replace_1_1.txt","r").read())

# masked_code_lines = open(f"/data/toli/State-Level-DP/mutation/subjects/exception/masked_CUT.txt","r").read()
# extracted_code = extract_code(open(f"/data/toli/State-Level-DP/mutation/subjects/exception/extracted_code.txt","r").read())

# refined_code = refine_extracted_code_for_mask_location(masked_code_lines.splitlines(), extracted_code.splitlines())
# print(f"refined_code:\n{refined_code}")
# resultant_code = snippet_infilling(masked_code_lines, refined_code)
# print(f"resultant_code:\n{resultant_code}")

# print(f"refined code:\n{snippet_infilling(masked_code_lines, extracted_code)}")

def extract_relevant_code_snippet_and_construct_complete_code(mask_start:int, mask_end, masked_CUT:str,
                                                                extracted_code:str):
    # print(f"!!!mask_location:{mask_location}")
    # print(f"!!!masked_CUT:{masked_CUT}")
    # print(f"!!!extracted_code:{extracted_code}")
    diff_object = diff(mask_start, mask_end, masked_CUT.splitlines(),extracted_code.splitlines())
    if diff_object != 0:
        return "\n".join(diff_object.C2)
    else:
        return None

def test_refined_extracted_code():
    # masked_code_lines = open(f"/data/toli/State-Level-DP/mutation/subjects/input/c1/simply_replace/evo1/evo1_{subject_num}_simply_replace_1.txt","r").readlines()
    # extracted_code = extract_code(open(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/prioritization/evo1/qwen/raw_output/evo1_{subject_num}_simply_replace_1_1.txt","r").read())
    masked_code_lines = open(f"/data/toli/State-Level-DP/mutation/subjects/input/c1/ast/evo27/evo27_15_ast_2_3_2.txt").readlines()
    extracted_code = extract_code(open(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/ast/evo27/starchat/raw_output/evo27_15_ast_2_3_2_1.txt","r").read())
    print(f"masked code:\n{('').join(masked_code_lines)}")
    print(f"extracted code:\n{extracted_code}")
    refined_extracted_code = refine_extracted_code_for_mask_location(masked_code_lines,extracted_code.splitlines())
    print(f"refined_extracted_code:\n{refined_extracted_code}") 
    # is_full_code(extracted_code, masked_code_lines,3-2+1)
    # infilled_extracted_code = snippet_infilling(masked_CUT, extracted_code_snippet_from_complete_code)

#has not problem
def test_is_full_code():
    extracted_code = extract_code(open(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/ast/evo27/starchat/raw_output/evo27_15_ast_2_3_2_1.txt","r").read())
    masked_code_lines = open("/data/toli/State-Level-DP/mutation/subjects/target_code/evo27.txt").read()
    print(is_full_code(extracted_code, masked_code_lines,3-2+1))

    

# test_is_full_code()
# test_refined_extracted_code()
# def extract_code(generated_code: str):
#     # print("Extracting code...")
#     extracted_lines = []
#     code_scope = False
#     has_code_backquote = False
#     for each_line in generated_code.splitlines():
#         if "<MASK>" in each_line:
#             continue
#         if "```" in each_line:
#             if not code_scope:
#                 code_scope = True
#                 has_code_backquote = True
#                 continue
#             else:
#                 break
#         if code_scope:
#             extracted_lines.append(each_line)
#     if not has_code_backquote:
#         return generated_code
#     return "\n".join(extracted_lines)

# #Handle the case that generated code contains lines exists in the original code, in this case, the redundant lines in generated code should be pruned.
# def refine_extracted_code_for_mask_location(masked_code_lines:list[str], extracted_code_lines:list[str]):
#     #closest_uppest_matched_line is the line of CUT that matches the line of extracted_code_lines and closes to function signature
#     closest_uppest_matched_extracted_code_line = -1
#     #closest_upppest_matched_line is the matched line of CUT furthest from the function signature
#     closest_lowest_matched_extracted_code_line = len(extracted_code_lines)

#     #Compute closest_upper_matched_line
#     matched_extracted_code_lines_idx = 0
#     tmp_uppest_matched_extracted_code_line = -1
#     for each_line_idx in range(len(masked_code_lines)):

#         #When hitting "<MASK>", flush the search result
#         if "<MASK>" in masked_code_lines[each_line_idx]:
#             closest_uppest_matched_extracted_code_line = tmp_uppest_matched_extracted_code_line
#             break

#         if masked_code_lines[each_line_idx].strip() in extracted_code_lines[matched_extracted_code_lines_idx]:
#             if matched_extracted_code_lines_idx > tmp_uppest_matched_extracted_code_line:
#                 tmp_uppest_matched_extracted_code_line = matched_extracted_code_lines_idx
#                 matched_extracted_code_lines_idx += 1 

#             #Handle the case that extracted_code is completely redunadant with code under test, at this case, just need to return empty string
#             if closest_uppest_matched_extracted_code_line == len(extracted_code_lines) - 1:
#                 return ""
#         else:
#             #When there is a line which the extracted line does not match cut before hitting <MASK>, the lines are just conincidentally matched (i.e., lines matched outside the mask location), hence tmp_uppest_matched_extracted_code_line is reset
#             tmp_uppest_matched_extracted_code_line = -1

            

#     #Compute closest_lowest_matched_line
#     matched_extracted_code_lines_idx = len(extracted_code_lines)-1
#     tmp_lowest_matched_extracted_code_line = len(extracted_code_lines)
#     #First reverse extracted code
#     # reversed_extracted_code_lines = list(reversed(extracted_code_lines))
#     for each_line_idx in list(reversed(range(len(masked_code_lines)))):

#         #When hitting "<MASK>", flush the search result
#         if "<MASK>" in masked_code_lines[each_line_idx]:
#             closest_lowest_matched_extracted_code_line = tmp_lowest_matched_extracted_code_line
#             break

#         if masked_code_lines[each_line_idx].strip() in extracted_code_lines[matched_extracted_code_lines_idx]:
#             print(f"Matched!!!each_line_idx is {each_line_idx}")
#             if matched_extracted_code_lines_idx < tmp_lowest_matched_extracted_code_line:
#                 #Note that it is impossible for closest_uppest_matched_line == closest_lowest_matched_line, because the first loop already handles the case that "extracted_code is completely redunadant" 
#                 tmp_lowest_matched_extracted_code_line = matched_extracted_code_lines_idx
#                 print(f"tmp_lowest_matched_extracted_code_line is updated:{tmp_lowest_matched_extracted_code_line}")
#                 matched_extracted_code_lines_idx -= 1
#         else:
#             tmp_lowest_matched_extracted_code_line = len(extracted_code_lines)

#     print(f"extracted_code_lines:{extracted_code_lines}")
#     print(f"closest_uppest_matched_extracted_code_line:{closest_lowest_matched_extracted_code_line}")

#     new_extracted_code = []
#     # if closest_uppest_matched_extracted_code_line != -1 or closest_lowest_matched_extracted_code_line != len(extracted_code_lines):
#         #Note that closest_lowest_matched_CUT_line - closest_uppest_matched_CUT_line >= 2 for sure, as the first loop already handles "extracted_code is completely redunadant" (i.e., the difference of the two <= 1)
#     for line_idx in range(closest_uppest_matched_extracted_code_line + 1, closest_lowest_matched_extracted_code_line): 
#         new_extracted_code.append(extracted_code_lines[line_idx])
#     # print(f"new_extracted_code:\n{new_extracted_code}")
#     return "\n".join(new_extracted_code)