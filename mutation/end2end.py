import socket
import sys
import os
here = os.path.dirname(__file__)
import shutil
sys.path.append(os.path.join(here, '..'))
from mutation.generate_code_with_llm import GenerateFuncWithLLM, Normal, constructGCLLM, GenerateClassWithLLM
import CrossVerificationDemo 
from mutation.utils import diff, list_txt_files, list_txt_files_without_subdir, get_target_code_path, \
    convert_test_case_to_dict
from semantic_analysis import Semantic_Analysis
from mutation.prioritization import evaluate_prioritization_quality,get_mask_block


class FunctionMode:
    def __init__(self, target):
        self.input_mode = "function"
        self.target = target

    def add_test_case(self, argument_list: list[str], test_case_text: str):
        test_case_list = test_case_text.splitlines()
        for test_case in test_case_list:
            argument_list.append(test_case)

    def read_target_signature(self):
        dict = construct_subject_functionSignature_dict()
        return dict[self.target]


class ClassMode:
    def __init__(self):
        self.input_mode = "class"

    def add_test_case(self, argument_list: list[str], test_case):
        argument_list.append(convert_test_case_to_dict(test_case.splitlines()))

    def read_target_signature(self):
        return None


def construct_subject_functionSignature_dict():
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


def test_get_input_starchat(target, input_mode, model):
    # if self.test_mode:
    func_sign_dict = construct_subject_functionSignature_dict()
    gcllm = constructGCLLM(target, model, "single", True, input_mode, Normal())
    if input_mode == "function":
        gcllm.input_mode_class = GenerateFuncWithLLM(target,func_sign_dict[target],model)
    elif input_mode == "class":
        gcllm.input_mode_class = GenerateClassWithLLM(gcllm)
    gcllm.initialise_hf_model()
    # target = "evo27"
    target_code_path = get_target_code_path(target)
    original_code = open(target_code_path,"r").read()
    docstring = f"/data/toli/State-Level-DP/mutation/subjects/prompt/{target}.txt"
    path_to_code_to_be_infilled = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}/{target}_div.txt"
    concatenated_content = (
            "Generate diverse inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
            + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    gcllm.generate_test_input(target_code_path, concatenated_content, path_to_code_to_be_infilled)
    concatenated_content = (
                    "Generate complex inputs for the target code. They should be different from example. There can be only one test case per line.\n\n## Docstring:\n"
                    + docstring + "## Target code\n```python\n" + original_code + "\n```"
    )
    path_to_code_to_be_infilled = f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}/{target}_com.txt"
    gcllm.generate_test_input(target_code_path, concatenated_content, path_to_code_to_be_infilled)

def txt_to_py(path:str):
    if not os.path.exists(path):
        assert os.path.exists(path.replace(".py",".txt"))
        shutil.copyfile(path.replace(".py",".txt"), path)

# def mutated_input():
#     print("Running mutated_input")
#     target = "evo58"
#     model = "starchat"
#     input_mode = "function"
#     target_signature_dict = construct_subject_functionSignature_dict()
#     test_input_file_list = list_txt_files(f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}")
#     assert len(test_input_file_list) > 0, f"No input files in /data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}"
#     sa_o = Semantic_Analysis("debug", target, input_mode, model, "linux")
#     for idx, each_test_input in enumerate(test_input_file_list[:2],1):
#         # if  "test_api" in each_test_input:
#         print(f"Working on idx:{idx}")
#         input_from_file = open(each_test_input, encoding="utf-8").read()
#         sa_o.mutate_input(input_from_file)
#         #!!!Reminder: execute_tests only accepts relative path (e.g., mutation/subjects/correct_code/evo12.py)
#
#         code1_path = f"mutation/subjects/correct_code/{target}.py"
#         code2_path = f"mutation/subjects/target_code/{target}.py"
#         txt_to_py(code1_path)
#         txt_to_py(code2_path)
#
#         result = sa_o.execute_tests([code1_path,code2_path],target, target_signature_dict[target],is_diff_testing=True)
#         print(f"Test execution result:{result}")
            # print(f"each_test_input:{each_test_input}")
'''
This function only performs mutation on input instead of generating new input using LLM
identifier is used to find file path of corresponding reference version.
The for loop ends when there are ten inconsistency revealing test, otherwise, it performs mutation for all inputs.
'''
def generate_incosistency_inducing_input(target:str, model, identifier:str,previous_test):
    if previous_test and len(previous_test) >= 10:
        print(f"Already obtained 10 failure-revealing tests!")
        return previous_test
    print("Running mutated_input")
    if "evo" in target:
        input_mode_class = FunctionMode(target)
    elif "classeval" in target:
        input_mode_class = ClassMode()
    else:
        raise Exception("Invalid target")
    reference_version_directory_path_list = f"mutation/subjects/output/c1/ast/{target}/{model}"
    
    test_input_file_list = list_txt_files(f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}")
    if len(test_input_file_list) == 0:
        test_get_input_starchat(target, input_mode_class.input_mode, model)
        test_input_file_list = list_txt_files(f"/data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}")
    assert len(test_input_file_list) > 0, f"No input files in /data/toli/State-Level-DP/mutation/subjects/test_input/{target}/{model}"
    sa_o = Semantic_Analysis("debug", target, input_mode_class.input_mode, model, "linux")
    
    #If there are failure-revealing tests found from other reference version, need to consider them
    resultant_argument_list = None
    if previous_test == None:
        resultant_argument_list = []
    else:
        resultant_argument_list = previous_test
    for idx, each_test_input in enumerate(test_input_file_list,1):
        # if  "test_api" in each_test_input:
        print(f"Working on idx:{idx}")
        input_from_file = open(each_test_input, encoding="utf-8").read()
        # sa_o.mutate_input(input_from_file)
        #!!!Reminder: execute_tests only accepts relative path (e.g., mutation/subjects/correct_code/evo12.py)

        #Use identifier to find reference version path list
        reference_version_path_list = [each_rv_path.replace(".txt",".py") for each_rv_path in list_txt_files_without_subdir(reference_version_directory_path_list) if identifier in each_rv_path]
        cut_path = get_target_code_path(target).replace(".txt", ".py")
        #RV are txt
        for each_rv in reference_version_path_list:
            txt_to_py(each_rv)
        txt_to_py(cut_path)
        for each_reference_version_path in reference_version_path_list:
            # print(f"each_reference_version_path:{each_reference_version_path}")
            txt_to_py(each_reference_version_path)
        print("Start executing tests...")
        #Note that resultant_argument_list has not included inputs generated by LLM, that should be included in execute_tests, in order to retrieve test that can increase coverage
        print(f"resultant_argument_list:{resultant_argument_list}")
        resultant_argument_list += sa_o.execute_tests(reference_version_path_list, target, input_mode_class.read_target_signature(),is_diff_testing=False,end2end=True,cut_path=cut_path,resultant_argument_list_arg=resultant_argument_list)
        print(f"Current status of resultant_argument_list:{resultant_argument_list}")
        print("Finish executing tests...")
        #If found over 10 inconsistency revealing test, no longer need to generate inputs
        if len(resultant_argument_list) >= 10:
            print(f"Already obtained 10 failure-revealing tests!")
            break
        else:
            print(f"Obtained {len(resultant_argument_list)} failure-revealing tests...")
    
    f = open(f"mutation/subjects/resultant_arguments/{target}.txt","w")
    f.write(str(resultant_argument_list))
    f.close()
    return resultant_argument_list
        # print(f"Test execution result:{result}")
            # print(f"each_test_input:{each_test_input}")

#subject=None,subject_type=None,model=None,verification_type=None,function_name=None,argument_list=None

# def pre_process_arg_str(arguments_list:str):
#     if "[["

def exercise_CrossVerificationDemo(target,model):
    # arguments_list = ['147, 247', '1000, 2000', '150, 250', '123456, 654321', '999999, 1000000', '2, 200000', '250000, 500000', '100, 10000', '987654, 1000000', '1, 99999', '100000, 200000', '1, 10000', '200, 500', '500, 1000', '5000, 10000', '7000, 8000', '2500, 7500', '300, 700', '8000, 12000', '9000, 15000', '100, 400', '123456, 654321', '987654, 1234567', '111111, 666666', '222222, 777777', '333333, 888888', '444444, 999999', '10000, 20000', '50000, 60000', '100000, 200000', '500, 1000', '1000, 2000', '100, 500', '5000, 10000', '100, 20000']
    function_name = None
    if "evo" in target:
        subject_type = FunctionMode(target)
        target_signature_dict = construct_subject_functionSignature_dict()
        function_name = target_signature_dict[target]
    elif "classeval" in target:
        subject_type = ClassMode()
    else:
        raise Exception("Invalid target")
    arguments_list = open(f"mutation/subjects/resultant_arguments/{target}.txt","r").read()
    if arguments_list == "[]":
        arguments_list = []
        test_case_file_list = list_txt_files(f"mutation/subjects/test_input/{target}/{model}")
        for each_test_file in test_case_file_list:
            with open(each_test_file,"r") as f:
                test_case_text = f.read()
                subject_type.add_test_case(arguments_list, test_case_text)
        arguments_list = str(arguments_list)
    verification_type="sd"
    print(f"Length of argument is {len(eval(arguments_list))}")
    CrossVerificationDemo.main(subject_arg=target,subject_type_arg=subject_type.input_mode,model_arg=model,verification_type_arg=verification_type,function_name_arg=function_name,argument_list_arg=str(arguments_list))

# def failing_test_selection():
#     arguments_list = open(f"resultant_argument_list.txt","r").read()


# def evaluation(target,target_block):
#     f =  open("/data/toli/State-Level-DP/output/evo72starchat_sd_g.txt","r")
#     encountered_target_block = False
#     test_result = {}
#     for each_line in f.readlines():
#         if target_block in each_line:
#             #the line consists only identifier and matches target_block
#             if target_block in each_line and "Input-Output Pair" not in each_line:
#                 print(f"encountered_target_block turns True")
#                 encountered_target_block = True
#                 continue
            
#             #the line consists only identifier and but does not match target_block
#             if not target_block in each_line and "Input-Output Pair" not in each_line:
#                 encountered_target_block = False
#                 continue

#             if encountered_target_block:
#                 # print(f"targeted each line:{each_line}")
#                 if "Input-Output Pair" in each_line:
#                     #Select the first test case collected
#                     result_type = each_line.split("[[")[1].split(",")[0]
#                     if "(not verified)" in each_line:
#                         continue
#                     if result_type not in test_result:
#                         test_result[result_type] = 1
#                     else:
#                         test_result[result_type] += 1

#     return test_result


def check_whether_inconsistency_revealing_input_is_found(result_dict):
    #Example format of result_dict:{'2_18(1,15)': {'Failure revealed!': 28, 'false positive': 1}, '15_16(1,15)': {'Failure revealed!': 8, 'false positive': 1}}
    for each_key in result_dict:
        print(f"Found inconsistency revealing test!")
        if result_dict[each_key]['Failure revealed!'] > 0 or result_dict[each_key]['false positive']:
            return True
    print(f"Inconsistency revealing test not found!")
    return False
        # print(f"result_dict[each_key]:{result_dict[each_key]}")


                    
def main(target_without_extension,model):
    os.makedirs("mutation/subjects/resultant_arguments", exist_ok=True)
    #for loop here, to iterate all prioritized block, until a inconsistency-inducing one is found
    # target_block = "15_16_2"
    # model = "starchat"
    using_generated_failing_input = True
    # how to select target block?
    result_dict = {}
    iteration = 0
    arguments_list = None
    #The while loop should handle the timeout problem?
    # while not check_whether_inconsistency_revealing_input_is_found(result_dict):
    prioritization_result = open(f'mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe/final.csv', 'r').read().splitlines()
    if len(prioritization_result) == 0:
        print("{}")
        return
    prioritization_result_to_analyze = prioritization_result[:5]
    # depth_0_block = prioritization_result[0]
    for idx, ranked_each_line in enumerate(prioritization_result_to_analyze,1):
        if arguments_list and len(arguments_list) >= 10:
            break
        else:
            print(f"Analyzing {idx}th prioritized line.")
        if len(ranked_each_line) > 0:
            prioritized_line = int(ranked_each_line.split(",")[0])
            involved_block = get_mask_block(target_without_extension,prioritized_line)
            involved_block.sort(key=lambda x: x[2],reverse = True)
            #The first block is always a full block
            for each_involved_block in involved_block[1:]:
                start_line = each_involved_block[0]
                end_line = each_involved_block[1]
                #Note that each_involved_block[2] is not depth, but difference in line number between start line and end line
                # depth = each_involved_block[2]
                # identifier is used to find file path of corresponding reference version
                identifier = f"{start_line}_{end_line}_"
                #No need to do arguments_list +=, because arguments_list will append inside this function and write to a file
                # assert arguments_list == None, "Forgot to comment out the following statement."
                argument_path = f"mutation/subjects/resultant_arguments/{target_without_extension}.txt"
                if os.path.exists(argument_path):
                    arguments_list = eval(open(argument_path,"r").read())
                if arguments_list and len(arguments_list) >= 10:
                    print(f"Already obtained 10 failure-revealing tests!")
                    break
                else:
                    arguments_list = generate_incosistency_inducing_input(target_without_extension, model, identifier,
                                                                          previous_test=arguments_list)
                    print(f"len(arguments_list) is {len(arguments_list)}")
                    iteration += 1
    exercise_CrossVerificationDemo(target_without_extension,model)
    result_dict = evaluate_prioritization_quality(target_without_extension,model,using_generated_failing_inpt_arg1=using_generated_failing_input)
    with open(
            f"mutation/subjects/output/c1/prioritization/{target_without_extension}/{model}/dataframe/final_output(generated_test).txt",
            "w", encoding="utf-8") as f:
        f.write(str(result_dict))
    print(f"result_dict is {result_dict}")


if __name__ == '__main__':
    subject_id_list = None
    if socket.gethostname() == "sccpu7.cse.ust.hk":
        # subject_id_list = #"65""72","76","78","84","12c","27c",

        subject_id_list = []
        #skipped: "36c","78","27c"
        #done: "12c","36c"，"39c","49c","55c","72c","75c","65","72","76","84"
    elif socket.gethostname() == "sccpu6.cse.ust.hk":
        subject_id_list = ["1","11","12","14","15","18","27","28","34","36","41","44","55","67","70","75","77","82","86","91","15c","27c","28c","34c","36c","41c","44c","49c","55c"]
        for idx, subject_id in enumerate(subject_id_list):
            subject_id_list[idx] = f"evo{subject_id}"
    elif socket.gethostname() == "sccpu8.cse.ust.hk":
        subject_id_list = ["48c","51c","64c","73c","81c","88c"]#"48","73","81","51","88","39","64"
    for subject_id in subject_id_list:
        model = "starchat"
        main(subject_id,model)



# arguments_list = generate_incosistency_inducing_input(target_without_extension,target_block)
# result_dict = evaluate_prioritization_quality(target_without_extension,model,using_generated_failing_inpt_arg1=using_generated_failing_inpt)
# result_dict = eval("{'2_18(1,15)': {'Failure revealed!': 28, 'false positive': 1}, '15_16(1,15)': {'Failure revealed!': 8, 'false positive': 1}, '15_15(1,15)': {'Failure revealed!': 0, 'false positive': 0}, '2_18(2,11)': {'Failure revealed!': 28, 'false positive': 1}, '11_12(2,11)': {'Failure revealed!': 36, 'false positive': 0}, '11_11(2,11)': {'Failure revealed!': 0, 'false positive': 0}, '2_18(3,18)': {'Failure revealed!': 28, 'false positive': 1}, '18_18(3,18)': {'Failure revealed!': 0, 'false positive': 0}, '2_18(4,12)': {'Failure revealed!': 28, 'false positive': 1}, '11_12(4,12)': {'Failure revealed!': 36, 'false positive': 0}, '12_12(4,12)': {'Failure revealed!': 0, 'false positive': 0}, '2_18(5,2)': {'Failure revealed!': 28, 'false positive': 1}, '2_2(5,2)': {'Failure revealed!': 43, 'false positive': 0}}")
# result_dict = eval("{'2_18(1,15)': {'Failure revealed!': 0, 'false positive': 0}, '15_16(1,15)': {'Failure revealed!': 1, 'false positive': 0}}")
# result_dict = {}
# print(check_whether_inconsistency_revealing_input_is_found(result_dict))
# print(arguments_list)
# print(f"arguments_list:{arguments_list}")
# Generate input and conduct evaluation
#
# exercise_CrossVerificationDemo()
# print(evaluation(target,target_block))