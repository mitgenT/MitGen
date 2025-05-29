import ast
from func_timeout import func_set_timeout
import json
import os
import pandas
import re
from Levenshtein import distance
from enum import IntEnum, auto
import difflib
from numpy import nan, inf
import sys
import copy
import statistics
import numpy as np
# sys.set_int_max_str_digits(0)

root = "mutation/subjects"

class Identifier:
    def __init__(self, depth, start, end):
        self.depth = depth
        self.start = start
        self.end = end

    def toString(self):
        return f"{self.start}_{self.end}_{self.depth}"


def calculate_min_depth(identifiers):
    min_depth = 99
    for identifier in identifiers:
        if identifier.depth.isnumeric() and int(identifier.depth) < min_depth:
            min_depth = int(identifier.depth)
    return min_depth


@func_set_timeout(120)
def class_execute(output_code, test_case_list: list[str], start_idx=0):
    class_name = get_class_name(output_code)
    try:
        exec(output_code, globals())
        output = []
        index = start_idx
        if f"{class_name}(" in test_case_list[0]:
            this_class = eval(test_case_list[0])
            test_case_list_without_init = test_case_list[1:]
        else:
            this_class = eval(f"{class_name}()")
            test_case_list_without_init = test_case_list
        for test_case in test_case_list_without_init:
            try:
                output.append([index, eval(f"this_class.{test_case}")])
            except AttributeError:
                output.append([index, eval(test_case)])
            index += 1
        globals().pop(class_name)
        return output, False
    except SyntaxError as e:
        globals().pop(class_name)
        return None, True
    except Exception as e:
        if class_name in globals():
            globals().pop(class_name)
        return None, True


def coarse_grain(subject, mode):
    """
    :return: depth|start|end|input filepath
    """
    os.makedirs("stub_folder", exist_ok=True)
    divided_group = []
    input_files = list_txt_files(f"{root}/input/c1/{mode}/{subject}")
    for input_file in input_files:
        c1_start = input_file.split("_")[-3]
        end_lineno = input_file.split("_")[-2]
        c1_depth = input_file.split("_")[-1].removesuffix(".txt")
        if c1_depth.isnumeric():
            c1_depth = int(c1_depth)
        divided_group.append({"depth": c1_depth, "start": c1_start, "end": end_lineno, "filepath": input_file})
    df = pandas.DataFrame(divided_group)
    df_numbers = df[df["depth"].apply(lambda x: isinstance(x, (int, float)))]
    df_special = df[df["depth"].apply(lambda x: isinstance(x, str))]
    df_numbers_sorted = df_numbers.sort_values(by=["depth", "start"], ascending=[True, True])
    df_special = df_special.sort_values(by="start", ascending=True)
    df_sorted = pandas.concat([df_numbers_sorted, df_special])
    return df_sorted


def convert_test_case_to_dict(test_case_list: list[str]):
    result = {}
    for test_case in test_case_list:
        try:
            index = test_case.index(":")
            test_case_split = [test_case[:index], test_case[index + 1:]]
            value = test_case_split[1]
            result[test_case_split[0]] = value
        except ValueError:
            result[test_case] = None
    return result


def create_identifiers(sorted_divided_group):
    '''
    :return: list of identifiers, Identifier(depth: string, start: integer, end: integer)
    '''
    identifiers = []
    for index, row in sorted_divided_group.iterrows():
        current_depth = str(row["depth"])
        current_start = int(row["start"])
        current_end = int(row["end"])
        identifiers.append(Identifier(current_depth, current_start, current_end))
    return identifiers


def cross_validation(sorted_divided_group):
    identifiers = create_identifiers(sorted_divided_group)
    identifier_group = {}
    min_depth = calculate_min_depth(identifiers)
    for identifier in identifiers:
        if identifier.end - identifier.start >= 1 and int(identifier.depth) == min_depth:
            temp_group = []
            for identifier_1 in identifiers:
                if (identifier_1.start >= identifier.start
                        and identifier_1.end <= identifier.end and identifier != identifier_1):
                    temp_group.append(identifier_1)
            identifier_group[identifier] = temp_group
    return identifier_group


def get_class_name(code):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name


def get_function_name(code):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name


def get_target_code_path(subject: str):
    if subject.endswith("c"):
        subject_without_suffix = subject.removesuffix('_' + subject.split("_")[-1])
        return f"{root}/correct_code/{subject_without_suffix}.txt"
    else:
        return f"{root}/target_code/{subject}.txt"


def initialisation(target_file:str,model:str, prompt_mode="full", output_directory=root):
    target_file_without_extension = target_file.replace(".txt","")
    print("Performing initialisation...")
    os.makedirs(f"{output_directory}/input/c2/add_specific/{target_file_without_extension}/{model}",exist_ok = True)
    os.makedirs(f"{output_directory}/input/c1/simply_replace/{target_file_without_extension}",exist_ok = True)
    os.makedirs(f"{output_directory}/input/c1/block_level/{target_file_without_extension}/{model}",exist_ok = True)
    os.makedirs(f"{output_directory}/input/c1/ast/{target_file_without_extension}",exist_ok = True)
    os.makedirs(f"executable",exist_ok = True)
    # os.makedirs(f"mutation/executable",exist_ok = True)
    if prompt_mode == "part":
        model = model + "_part"
    os.makedirs(f"{output_directory}/output/c2/add_specific/{target_file_without_extension}/{model}",
                exist_ok=True)
    os.makedirs(f"{output_directory}/output/c1/simply_replace/{target_file_without_extension}/{model}",
                exist_ok=True)
    os.makedirs(f"{output_directory}/output/c1/block_level/{target_file_without_extension}/{model}", exist_ok=True)
    os.makedirs(f"{output_directory}/output/c1/ast/{target_file_without_extension}/{model}", exist_ok=True)
    os.makedirs(f"{output_directory}/output/c1/ast/{target_file_without_extension}/{model}/raw_output", exist_ok=True)
    os.makedirs(f"{output_directory}/time", exist_ok=True)
        # os.makedirs(f"{output_directory}/{in_or_out}/c1/simply_insert/{target_file_without_extension}/{model}",exist_ok = True)
    

def get_indent(line:str) -> str:
    indent = len(line) - len(line.lstrip())
    return " " * indent

def get_indent_for_block(code:str,start_line:str,end_line:str) -> str:
    all_indents = []
    for each_line in range(start_line,end_line+1):
        indent = len(code[each_line]) - len(code[each_line].lstrip())
        all_indents.append(indent)
    return " " * min(all_indents)


def list_txt_files(directory:str) -> list[str]:
    """
    :return: txt files with full path.
    """
    # print("!!!list_txt_files is recursive. Please manually exclude files that are not related.")
    txt_files = []
    for filename in os.listdir(directory):
        #We exclude backup files from analysis
        if "backup" in filename:
            continue
        # print(f"filename:{filename}")
        filepath = f"{directory}/{filename}"
        if os.path.isdir(filepath):
            txt_files += list_txt_files(filepath)
        elif filename.endswith('.txt'):
            txt_files.append(filepath)
    return txt_files


def list_txt_files_without_subdir(directory:str) -> list[str]:
    txt_files = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = f"{directory}/{filename}"
                if os.path.isfile(filepath):
                    txt_files.append(filepath)
    except FileNotFoundError:
        print(f"{directory} not found")
        return txt_files
    return txt_files


def load_args(subject):
    with open(f"mutation/subjects/args/{subject}.json", "r") as f:
        data = json.load(f)
    return data


def file_check(target:str):
    print(f"!!!target:{target}!!!")
    subject = target.removesuffix(".txt")
    assert os.path.isfile(get_target_code_path(subject)), f"{get_target_code_path(subject)} is missing"
    if "classeval" in target:
        filename_of_docstring = '_'.join(target.split("_")[:2]) + ".txt"
        assert os.path.isfile(f"{root}/prompt/{filename_of_docstring}"), f"{root}/prompt/{filename_of_docstring} is missing"
    else:
        assert os.path.isfile(f"{root}/prompt/{target}"), f"{root}/prompt/{target} is missing"


def is_same_syntax_by_ast(tree1, tree2):
    return distance(ast.dump(tree1), ast.dump(tree2)) < 1


def pass_test_case(func, args=""):
    match args:
        case "":
            return eval(f"{func}()")
        case _:
            return eval(f"{func}({args})")


def wrap_as_function(code_lines, indent, mode):
    """
    :param code_lines:
    :param indent: number of space
    :param mode:
    :return:
    """
    indent_space = ' ' * indent
    new_code_lines = [indent_space + line for line in code_lines]
    if mode == "stub":
        new_code_lines = (
            ["ic = -1", "def quest(input_list, diction):", indent_space + "global ic",indent_space + "ic = -1"]
            + new_code_lines + [indent_space + "return diction"]
            + ["def get_input(input_list):  # pragma: no cover", indent_space + "global ic", indent_space + "ic +=1",
            indent_space + "return input_list[ic]"]
        )
    else:
        new_code_lines = (
                ["ic = -1", "def quest(input_list):", indent_space + "global ic", indent_space + "ic = -1"]
                + new_code_lines
                + ["def get_input(input_list):  # pragma: no cover", indent_space + "global ic",
                   indent_space + "ic +=1",
                   indent_space + "return input_list[ic]"]
        )
    for i, item in enumerate(new_code_lines):
        new_code_lines[i] = (
            re.sub(r"input\s*\(\)", "get_input(input_list)", item)
            .replace("sys.stdin.readline()","get_input(input_list)")
            .replace("stdin.readline()", "get_input(input_list)")
        )
    return new_code_lines


def analyze_indentation(code: str):
    lines = code.splitlines()

    indent = 4
    for line_number, line in enumerate(lines, start=1):
        # 跳过空行
        if not line.strip():
            continue

        # 计算行首的空格和制表符总数
        indent_chars = len(line) - len(line.lstrip(' \t'))
        # 检查是否是4个空格的倍数
        if indent_chars % 4 != 0:
            indent = indent_chars
            break
    return indent


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    # TODO: search for any close floats? (other data structures)
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


class DataType(IntEnum):
    Float = auto()
    Bool = auto()
    Int = auto()
    Str = auto()
    Null = auto()
    Tuple = auto()
    List = auto()
    Dict = auto()
    Set = auto()
    Type = auto()
    Unknown = auto()


def get_type(x):
    if x is None:
        return DataType.Null
    elif isinstance(x, bool):
        return DataType.Bool
    elif isinstance(x, int):
        return DataType.Int
    elif isinstance(x, str):
        return DataType.Str
    elif is_floats(x):
        return DataType.Float
    elif isinstance(x, tuple):
        return DataType.Tuple
    elif isinstance(x, list):
        return DataType.List
    elif isinstance(x, dict):
        return DataType.Dict
    elif isinstance(x, set):
        return DataType.Set
    elif isinstance(x, type):
        return DataType.Type
    else:
        return DataType.Unknown


def is_equal(x, y) -> tuple[bool, str]:
    x_type, y_type = get_type(x), get_type(y)
    if x_type != y_type:
        return False, "Type mismatch: {} vs {}".format(str(x_type), str(y_type))

    if x_type in [DataType.Int, DataType.Bool, DataType.Null, DataType.Str, DataType.Set, DataType.Type]:
        if x == y:
            return True, None
        try:
            error_msg = "INT/BOOL/NULL/ Value mismatch: {} vs {}".format(repr(x)[:300], repr(y)[:300])
        except:
            error_msg = "Value mismatch: too large for display"
        return False, error_msg
    elif x_type == DataType.Float:
        try:
            if np.allclose(x, y, equal_nan=True, atol=1e-6): # guard against nan
                return True, None
            else:
                return False, "FLOAT Value mismatch: {} vs {}".format(x, y)
        except ValueError:
            if np.array_equal(x, y): return True, None
            else: return False, "FLOAT Value mismatch: {} vs {}".format(x, y)
    elif x_type in [DataType.List, DataType.Tuple]:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for i in range(len(x)):
            try:
                equal, msg = is_equal(x[i], y[i])
            except ValueError:
                return False, "Shape mismatch: {} vs {}".format(x[i], y[i])
            if not equal:
                return False, msg
        return True, None
    elif x_type == DataType.Dict:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for k, v in x.items():
            if k not in y:
                return False, "DICT Value mismatch: key {} in {} but not in {}".format(k, x, y)
            equal, msg = is_equal(v, y[k])
            if not equal:
                return False, msg
        return True, None
    else:
        try:
            if x == y:  # e.g., object comparison
                return True, None
            else:
                return False, "ELSE Value mismatch: {} vs {}".format(x, y)
        except:
            return False, "Unsupported type: {} <-- {}".format(x_type, type(x))
        


def check_output(result_file_path:str):
    f = open(result_file_path,'r')
    correct = None
    buggy = None
    base = {}
    for each_line in f.readlines():
        if 'correctCode:::' in each_line:
            input_list = eval(each_line.split(":::")[1])
            correct = [i[1] for i in input_list]
        elif 'buggyCode:::' in each_line:
            input_list = eval(each_line.split(":::")[1])
            buggy = [i[1] for i in input_list]
        elif 'baseCode' in each_line:
            key = each_line.split(":")[0].replace("baseCode","")
            base[key] = eval(each_line.split(":::")[1])
    # print(f"correct:{correct}")
    print(base)
    cross_check(base,correct,buggy)


    # print(correct)

def cross_check(base,correct,buggy):
    total_num_outputs = len(correct)
    baseKeys = list(base.keys())
    pairwise_match = {}
    pairwise_match_fp = {}
    pairwise_match_reveal_fault = {}
    pairwise_match_correct = {}
    for eachRvIdx in range(0,len(baseKeys)):
        RvOut = [i[1] for i in base[baseKeys[eachRvIdx]]]
        lenOfOutputs = len(RvOut)
        pairwise_match[baseKeys[eachRvIdx]] = {}
        pairwise_match_fp[baseKeys[eachRvIdx]] = {}
        pairwise_match_reveal_fault[baseKeys[eachRvIdx]] = {}
        pairwise_match_correct[baseKeys[eachRvIdx]] = {}
        # for toComapreRvIdx in range(eachRvIdx+1,len(baseKeys)):

        pairwise_match[baseKeys[eachRvIdx]] = []
        pairwise_match_fp[baseKeys[eachRvIdx]] = []
        pairwise_match_reveal_fault[baseKeys[eachRvIdx]] = []
        pairwise_match_correct[baseKeys[eachRvIdx]] = []
        for k in range(lenOfOutputs):
            match_correct, _ = is_equal(RvOut[k], correct[k])
            match_buggy, _ = is_equal(RvOut[k], buggy[k])
            pairwise_match[baseKeys[eachRvIdx]].append(k)
            if match_correct:
                pairwise_match_correct[baseKeys[eachRvIdx]].append(k)
                if not match_buggy:
                    pairwise_match_reveal_fault[baseKeys[eachRvIdx]].append(k)
            if not match_correct and not match_buggy:
                # print("=========Not match=========")
                # print(f"not matched, {RvOut[k]} Vs {correct[k]}")
                pairwise_match_fp[baseKeys[eachRvIdx]].append(k)


    print("============correct==========")
    print(pairwise_match_correct)
    pair_correct = []
    for eachKey in pairwise_match_correct:
        # for eachCompareKey in pairwise_match_correct[eachKey]:
        if len(pairwise_match_correct[eachKey]) == total_num_outputs:
            pair_correct.append((eachKey,len(pairwise_match_correct[eachKey])))    
    print("============Failure revealing==========")
    print(pairwise_match_reveal_fault)
    pair_reveal_fault = []
    for eachKey in pairwise_match_reveal_fault:
        # for eachCompareKey in pairwise_match_reveal_fault[eachKey]:
        if len(pairwise_match_reveal_fault[eachKey]) > 0:
            pair_reveal_fault.append((eachKey,len(pairwise_match_reveal_fault[eachKey])))
    print("============False positives==========")
    print(pairwise_match_fp)
    pair_fp = []
    for eachKey in pairwise_match_fp:
        # for eachCompareKey in pairwise_match_fp[eachKey]:
        if len(pairwise_match_fp[eachKey]) > 0:
            pair_fp.append((eachKey,len(pairwise_match_fp[eachKey])))
    print(f"*****Pairs that correct: {pair_reveal_fault}")
    print(f"*****Pairs that reveal faults: {pair_reveal_fault}")
    print(f"*****Pairs that returns fp: {pair_fp}")

class BlockPair:
    locations = [[], []]
    C1 = []
    C2 = []

    def __init__(self):
        self.locations = [[], []]  # Example: [[2, 3, 4, 5], [3, 4, 5]]
        self.C1 = []
        self.C2 = []

def reset_diff_code(code, diff_range):
    """
    Update code list according to the new diff range.
    :param code: original code list of C/C'
    :return: updated code list
    """
    result = []
    for lineno in diff_range:
        result.append(code[lineno - 1])
    if len(diff_range) > 0:
        if "break" in code[diff_range[-1] - 1]:
            result[-1] = result[-1].replace("break", "pass")
    # for i, line in enumerate(result):
    #     if "break" in line:
    #         result[i] = line.replace("break", "pass")
    return result


def diff(mask_start, mask_end, str1_lines, str2_lines):
    """
    This function output differing lines between cut and generated code.

    :param mask_start,mask_end: start line and end line of mask block that we are interested, to filter lines that we are not interested in
    :returns: 0 if no differing lines are found, otherwise, return the blockpair, not that blockpair.locations' first element is differing lines of C1, and second element is the differing lines of C2
    :raises keyError: raises an exception
    """
    mask_block = [mask_start, mask_end]
    for i in range(len(str1_lines) - 1, -1, -1):
        if not str1_lines[i]:
            str1_lines.pop(i)
            if mask_start > i + 1:
                mask_block[0] -= 1
            if mask_end > i + 1:
                mask_block[1] -= 1
    diffs = list(difflib.ndiff(str1_lines, str2_lines))
    # print(f"diffs:{diffs}")
    # for each_line in diffs:
    #     print(each_line)
    first_location = 1
    second_location = 1
    pairs = []
    current_pair = copy.deepcopy(BlockPair())
    for i in range(len(diffs)):
        line = diffs[i]
        if line.startswith(' '):
            first_location += 1
            second_location += 1
        elif line.startswith('-'):
            current_pair.locations[0].append(first_location)
            current_pair.C1.append(line)
            first_location += 1
        elif line.startswith('+'):
            current_pair.locations[1].append(second_location)
            current_pair.C2.append(line)
            second_location += 1

        if line.startswith(' ') and i != 0:
            if diffs[i - 1].startswith('-') or diffs[i - 1].startswith('+') or diffs[i - 1].startswith('?'):
                pairs.append(current_pair)
                current_pair = copy.deepcopy(BlockPair())
    if current_pair.locations != [[], []]:  # 增加位于代码最后的BlockPair
        pairs.append(copy.deepcopy(current_pair))

    result = copy.deepcopy(BlockPair())
    for blockpair in pairs:
        if blockpair.locations[0]:
            # print(f"!!!!!mask_block:{mask_block}")
            # print(f"blockpair.locations:{blockpair.locations}")
            if blockpair.locations[0][0] >= mask_block[0] and blockpair.locations[0][-1] <= mask_block[1]:
                result.locations[0] += blockpair.locations[0]
                result.locations[1] += blockpair.locations[1]
    if result.locations[0] and result.locations[1]:
        min_lineno_c1 = result.locations[0][0]
        max_lineno_c1 = result.locations[0][-1]
        min_lineno_c2 = result.locations[1][0]
        max_lineno_c2 = result.locations[1][-1]
        for lineno in range(min_lineno_c1, max_lineno_c1 + 1):
            if lineno not in result.locations[0]:
                result.locations[0].append(lineno)
        for lineno in range(min_lineno_c2, max_lineno_c2 + 1):
            if lineno not in result.locations[1]:
                result.locations[1].append(lineno)
        result.locations[0].sort()
        result.locations[1].sort()
        result.C1 = reset_diff_code(str1_lines, result.locations[0])
        result.C2 = reset_diff_code(str2_lines, result.locations[1])
        return result

    min_distance = 9999  # distance最小意味着最靠近mask location
    result = 0
    for blockpair in pairs:
        for i in range(len(blockpair.C1)):
            blockpair.C1[i] = blockpair.C1[i].removeprefix("- ")
            blockpair.C1[i] = blockpair.C1[i].removeprefix("+ ")
        for i in range(len(blockpair.C2)):
            blockpair.C2[i] = blockpair.C2[i].removeprefix("- ")
            blockpair.C2[i] = blockpair.C2[i].removeprefix("+ ")

        if compare_ignore_space(blockpair.C1, blockpair.C2):
            continue
        if len(blockpair.locations[0]) >= 1:
            # print(f"!!!!!mask_block:{mask_block}")
            # print(f"blockpair.locations:{blockpair.locations}")
            if (not has_overlapping_lines(mask_block, blockpair.locations[0]) and
                    (abs(blockpair.locations[0][-1] - mask_block[0]) > 1 or abs(
                    blockpair.locations[0][0] - mask_block[1]) > 1)):
                continue
            else:
                linenum1 = statistics.mean(blockpair.locations[0])
        if len(blockpair.locations[1]) >= 1:
            if (not has_overlapping_lines(mask_block, blockpair.locations[1]) and
                    (abs(blockpair.locations[1][-1] - mask_block[0]) > 1 or abs(
                    blockpair.locations[1][0] - mask_block[1]) > 1)):
                continue
            else:
                linenum2 = statistics.mean(blockpair.locations[1])
        if len(blockpair.locations[0]) == 0:
            linenum1 = linenum2
        elif len(blockpair.locations[1]) == 0:
            linenum2 = linenum1
        mean_mask_location = statistics.mean(mask_block)
        distance = abs(mean_mask_location - linenum1) + abs(mean_mask_location - linenum2)
        if distance < min_distance:
            min_distance = distance
            result = blockpair
    return result

def compare_ignore_space(code_list1, code_list2):
    code1 = "\n".join(code_list1)
    code2 = "\n".join(code_list2)
    code1_nospace = code1.replace(" ", "")
    code2_nospace = code2.replace(" ", "")
    return code1_nospace == code2_nospace


def has_overlapping_lines(mask_block, location):
    list1 = []
    for i in range(mask_block[0], mask_block[-1] + 1):
        list1.append(i)
    return bool(set(list1) & set(location))


def test_diff():
    mask_location = 20
    target_code_lines = open("mutation/subjects/target_code/classeval_assessmentsystem_7.txt", "r").readlines()
    generated_code = open(
        f"mutation/subjects/output/c1/prioritization/classeval_assessmentsystem_7/qwen/classeval_assessmentsystem_7_{mask_location}_simply_replace_1_1.txt",
        "r").readlines()
    diff_object = diff(mask_location,target_code_lines,generated_code)
    # print(diff_object)
    print(diff_object.locations)
    print(diff_object.C1)
    print(diff_object.C2)


def get_line_num(output_file_name,mode):
    starting_line_num = None
    ending_line_num = None
    if mode == "simply_apply" or mode == "simply_replace":
        if "classeval" in output_file_name:
            #The third index is subject ID
            starting_line_num = output_file_name.split("_")[3]
            ending_line_num = starting_line_num
        elif "evo" in output_file_name:
            #The second index is subject ID
            starting_line_num = output_file_name.split("_")[1]
            ending_line_num = starting_line_num
        else:
            raise ValueError(f"Invalid line number: {output_file_name}")
    elif mode == "ast":
        if "classeval" in output_file_name:
            # print(f"!!!target:{output_file_name}")
            assert False, "To implement"
            starting_line_num = output_file_name.split("_")[3]
            ending_line_num = output_file_name.split("_")[4]
        elif "evo" in output_file_name:
            starting_line_num = output_file_name.split("_")[3]
            ending_line_num = output_file_name.split("_")[4]
        else:
            raise ValueError("Invalid line number")
    else:
        raise ValueError("Invalid mode")
    return starting_line_num,ending_line_num

def remove_empty_line_and_comment_from_code(code:str):
    #first remove comment, because after removal there may be empty lines
    resultant_code_lines = []
    for each_line in code.splitlines():
        if not each_line:
            continue
        elif len(each_line.strip()) == 0:
            continue
        else:
            #Qwen may add print to the code
            # if "print(" in each_line:
            #     continue

            if len(each_line.strip()) > 0 and each_line.strip()[0] == "#":
                continue

            #comment in the line
            if "#" in each_line:
                comment_symbol_index = each_line.index("#")
                resultant_code_lines.append(each_line[:comment_symbol_index])
                continue

            resultant_code_lines.append(each_line)
    # print("=========Refined code========")
    # print("\n".join(resultant_code_lines))
    return "\n".join(resultant_code_lines)
    
if __name__ == "__main__":
    test_diff()