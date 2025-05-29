import csv
import json
import os.path
import time
from datetime import timedelta

import CrossVerificationDemo
from mutation import generate_code_with_llm, end2end, perform_mutation
from mutation.prioritization import main_auto, evaluate_prioritization_quality
from mutation.utils import load_args


def automate_prioritization(dataset_list: list[str], subject_dict: dict, model: str, generate_result_text: bool):
    if generate_result_text:
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
    else:
        main_auto(dataset_list, subject_dict, True, model)


def change_mask_location(subject, model):
    mask_location_list = []
    with open(f"mutation/subjects/output/c1/prioritization/{subject}/{model}/dataframe/final.csv", "r", encoding="utf-8") as f:
        f = csv.reader(f)
        for row in f:
            mask_location_list.append(int(row[0]))
    with open(f"mutation/subjects/args/{subject}.json", "r", encoding="utf-8") as f:
        json_text = f.read()
    args_dict = json.loads(json_text)
    if not mask_location_list:
        mask_location_list = None
    args_dict["mask_location"] = mask_location_list
    with open(f"mutation/subjects/args/{subject}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(args_dict))
    return mask_location_list


def create_args_json_for_classeval(dataset_list, subject_dict):
    json_text = '{"mask_location": null, "subject_type": "class"}'
    for each_dataset in dataset_list:
        if each_dataset == "evoeval":
            json_text = '{"mask_location": null, "subject_type": "function"}'
        for each_subject_id in subject_dict[each_dataset]:
            file_path = f"mutation/subjects/args/{each_dataset}_{each_subject_id}.json"
            if not os.path.isfile(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_text)


if __name__ == '__main__':
    model = "llama3-perplexity"
    dataset_list = []  # e.g. "evoeval", "classeval_xxx"
    subject_dict = {}  # e.g. "evoeval":['xx'], "classeval_xxx": ['xx']
    create_args_json_for_classeval(dataset_list, subject_dict)
    print("Starting prioritization...")
    automate_prioritization(dataset_list, subject_dict, model, False)
    for key, value in subject_dict.items():
        if key == "evoeval":
            for subject_id in value:
                subject = f"evo{subject_id}"
                target = f"{subject}.txt"
                mask_location_list = change_mask_location(subject, model)
                args = load_args(target.removesuffix(".txt"))
                func_name = args["func_name"]
                perform_mutation.main(target, model, func_name, "ast", "function")
                print(f"Generating {subject} code...")
                start_time = time.time()
                generate_code_with_llm.main(target, model, "ast", False, func_name, mask_location_list)
                elapsed_time = time.time() - start_time
                with open(f"mutation/subjects/time/{model}_{target}", "a", encoding="utf-8") as time_file:
                    time_file.write(f'elapsed time: {timedelta(seconds=elapsed_time)}\n')
                print(f"Generating {subject} data...")
                CrossVerificationDemo.main(subject, "function", model, "cvd")
                CrossVerificationDemo.main(subject, "function", model, "sd")
                CrossVerificationDemo.main(subject, "function", model, "dp")
        elif "classeval" in key:
            for subject_id in value:
                subject = f"{key}_{subject_id}"
                target = f"{subject}.txt"
                perform_mutation.main(target, model, None, "ast", "class")
                mask_location_list = change_mask_location(subject, model)
                print(f"Generating {subject} code...")
                start_time = time.time()
                generate_code_with_llm.main(target, model, "ast", False, mask_location=mask_location_list)
                elapsed_time = time.time() - start_time
                with open(f"mutation/subjects/time/{model}_{target}", "a", encoding="utf-8") as time_file:
                    time_file.write(f'elapsed time: {timedelta(seconds=elapsed_time)}\n')
                print(f"Generating {subject} data...")
                CrossVerificationDemo.main(subject, "class", model, "cvd")
                CrossVerificationDemo.main(subject, "class", model, "sd")
                CrossVerificationDemo.main(subject, "class", model, "dp")
    print("Getting prioritization data...")
    automate_prioritization(dataset_list, subject_dict, model, True)
    print("Performing end2end...")
    for key, value in subject_dict.items():
        if key == "evoeval":
            for subject_id in value:
                subject = f"evo{subject_id}"
                end2end.main(subject, model)
        elif "classeval" in key:
            for subject_id in value:
                subject = f"{key}_{subject_id}"
                print(f"end2end: {subject}")
                end2end.main(subject, model)