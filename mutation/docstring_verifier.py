from prioritization import construct_gllm,derive_input_mode,construct_subject_name_including_txt,construct_paths_to_code_to_be_infilled,constructing_concatenated_content,find_line_of_first_def,construct_gllm
import sys
import os
sys.path.append("/data/toli/State-Level-DP/mutation")
import generate_code_with_llm as gcllm
from codebleu import calc_codebleu

 
def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
     
    return intersection / union

# import sys 
# sys.path.append(os.path.join(here, '..'))

def verify_dcostring_similarity(refined_docstring,docstring_list):
    for each_docstring_idx in docstring_list:
        similarity_score = measure_docstring_difference(refined_docstring,docstring_list[each_docstring_idx])
        if similarity_score >= 0.8:
            return False
    return True

def prompt_docstring(gcllm_o,target_subject_file_name,model,retry_time_for_generating_docstring):
    instruction = "Paraphrase the following docstring. Output only the Paraphrased docstring."
    target_docstring = open(f"/data/toli/State-Level-DP/mutation/subjects/prompt/{target_subject_file_name}.txt").read()
    concatenated_content = f"{instruction}\n{target_docstring}"
    refined_docstring = gcllm_o.prompt_llm(concatenated_content)
    docstring_path = f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{target_subject_file_name}/{model}/{retry_time_for_generating_docstring}/docstring/docstring.txt"
    os.makedirs(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{target_subject_file_name}/{model}/{retry_time_for_generating_docstring}/docstring", exist_ok=True)
    # assert os.path.exists("cs.log") == False, f"Docstring already exists:{os.path.dirname(output_file_path)}/docstring/docstring.txt"
    if refined_docstring != None:
        with open(docstring_path, "w", encoding="utf-8") as output_file:
            output_file.write(refined_docstring)
    return refined_docstring

def generate_docstring(gcllm_o,target_subject_file_name,model,retry_time_for_generating_docstring,skip_already_exist):
    docstring_list = {}
    
    for ith_generated_docstring in range(1, retry_time_for_generating_docstring + 1):    
        # output_path = f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{target_subject_file_name}/{model}/{retry_time_for_generating_docstring}/docstring/docstring.txt"
        attempt = 0
        while True:
            if skip_already_exist == True:
                docstring_list[ith_generated_docstring] = open(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{target_subject_file_name}/{model}/{retry_time_for_generating_docstring}/docstring/docstring.txt","r").read()
                break
            refined_docstring = prompt_docstring(gcllm_o,target_subject_file_name,model,ith_generated_docstring)
            if refined_docstring != None and verify_dcostring_similarity(refined_docstring,docstring_list):
                docstring_list[ith_generated_docstring] = refined_docstring
                break
            else:
                attempt += 1
            if attempt == 10:
                break
    return docstring_list

def generate_code_with_paraphrased_docstring():
    #Parameters here
    print(f"Starting...")
    model = sys.argv[3]
    target_dataset = "evoeval"
    retry_time_for_generating_docstring = 10
    retry_time_for_generating_code = 10
    # code_to_be_infilled_to_verify = "/data/toli/State-Level-DP/mutation/subjects/input/c1/simply_replace/classeval_assessmentsystem_5/classeval_assessmentsystem_5_12_simply_replace_1.txt"
    each_target_id = sys.argv[1]
    total_line = sys.argv[2]
    subject_name = f"evo{each_target_id}"
    target_code_to_be_infilled = f"evo{each_target_id}_{total_line}_ast_3_3_s"
    skip_already_exist = False
    if skip_already_exist == False and os.path.exists(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{subject_name}/{model}"):
        assert False, "skip_already_exist set to False while identical code exists, delete existing code first"
    input_mode = derive_input_mode(target_dataset)
    print(f"Parameters Initiated...")
    # for each_target_id in target_id_list:
    target_subject_file_name = construct_subject_name_including_txt(target_dataset,each_target_id)
    print(f"target_subject_file_name:{target_subject_file_name}")
    input_file_path_list = construct_paths_to_code_to_be_infilled(target_subject_file_name,"ast")
    # print(f"input_file_path_list:{input_file_path_list }")
    gcllm_o = construct_gllm(model,input_mode,skip_already_exist)

    docstring_list = generate_docstring(gcllm_o,target_subject_file_name.replace(".txt",""),model,retry_time_for_generating_docstring,skip_already_exist)
    # generate_docstring(retry_time_for_generating_docstring)

    for each_code_to_be_infilled_path in input_file_path_list:

        #Debug
        # if "7_7_4" not in each_code_to_be_infilled_path:
        #     continue
        # Match particular each_code_to_be_infilled_path
        # assert False, "no need to generate repeated input for each docstring"
        if True:
        # if target_code_to_be_infilled in each_code_to_be_infilled_path:
            gcllm_o.update_target(target_subject_file_name)
            for ith_generated_docstring in docstring_list:
                #Debug
                # if ith_generated_docstring != 1:
                #     continue
                refined_docstring = docstring_list[ith_generated_docstring]
                # print(f"Docstring trial:{ith_generated_docstring}")
                # output_path_for_docstring = gcllm_o.creating_output_code_path(each_code_to_be_infilled_path, 0, prioritisation=False, docstring_verifier=True, generated_docstring_index=ith_generated_docstring)
                # refined_docstring = open(f"/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/{target_subject_file_name}/{model}/{retry_time_for_generating_docstring}/docstring/docstring.txt","r").read()
                for ith_generated_generated_code in range(1, retry_time_for_generating_code + 1):
                    #Generate a complete path for ith generated code
                    #output_file_path_specification: {self.subject}/{self.model}/{generated_docstring_index}/output_file_name
                    output_file_path = gcllm_o.creating_output_code_path(each_code_to_be_infilled_path, ith_generated_generated_code, prioritisation=False, docstring_verifier=True, generated_docstring_index=ith_generated_docstring)
                    # print(f"!!!output_file_path is:{output_file_path}")
                    # assert False, "debugging"
                    masked_CUT = None
                    with open(each_code_to_be_infilled_path, 'r', encoding="utf-8") as file2:
                        masked_CUT = file2.read()
                    function_signature_line_idx = find_line_of_first_def(masked_CUT)
                    actual_task_prompt_instruction = constructing_concatenated_content(target_subject_file_name,masked_CUT,function_signature_line_idx,refined_docstring)
                    #TODO: modify the postfix of generated code 
                    #Need to manually add output path, otherwise the inner function will call creating_output_code_path again and the original output path will be created
                    # gcllm.(target, model, mode, input_mode)
                    print(f"!!!Generated output path is:{output_file_path}")
                    gcllm_o.output_code_path = output_file_path
                    if target_dataset == "evoeval":
                        gcllm_o.input_mode_class = gcllm.GenerateFuncWithLLM(target_subject_file_name)
                    else:
                        raise ValueError("Unhandle verify class.")
                    gcllm_o.generate_infilled_code_snippets(masked_CUT, f'mutation/subjects/target_code/{target_subject_file_name}', each_code_to_be_infilled_path, 1,
                                                actual_task_prompt_instruction, 1.0, False, skip_already_exist)


def measure_docstring_difference(docstring1,docstring2):
    res = jaccard_similarity(set(docstring1.split(" ")), set(docstring2.split(" ")))
    return res


#==========Generating docstring=========
print("Start running!!")
generate_code_with_paraphrased_docstring()
print("Finish!!")

#==========Copmute docstring diff============
# docstring1 = open("/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/evo15_backup/qwen/1/docstring/dv_1_evo15_8_ast_2_2_2_0.txt","r").read()
# docstring2 = open("/data/toli/State-Level-DP/mutation/subjects/output/c1/docstring_verifier/evo15_backup/qwen/1/docstring/dv_1_evo15_8_ast_3_3_s_0.txt","r").read()
# docstring1 = "a = 2"
# docstring2 = "a = 1"
# print(measure_docstring_difference(docstring1,docstring2))
# print(output['codebleu'])
# print(jaccard_similarity(set(docstring1.split(" ")), set(docstring2.split(" "))))