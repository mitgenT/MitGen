'''
This program performs mutation.
'''
import ast
import sys
import os
here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))
# sys.path.append(os.path.abspath('..'))  
from mutation.utils import get_indent, initialisation, file_check, root, list_txt_files, get_target_code_path
from mutation.mutation_operator import define_specific_mutation, simply_replace_missing_line
from mutation.ast_mutator import AstMutator

#If there is any line no need to be masked, add the keyword to the following list.
keywords_cannot_any_mutation = ["def","import","try","except","finally","class"]

keywords_cannot_perform_specific_mutation = ["if","elif","else",]

# mutation_operators = [\
#     "add_specific(each_line,indent,each_option,keywords_cannot_perform_specific_mutation)",\
#     "simply_replace_missing_line(each_line,indent)",\
#     "simply_insert_missing_line(each_line,indent)",\
#     ]

# mutation_operators = [\
#     "add_specific(each_line,indent,each_option,keywords_cannot_perform_specific_mutation)",\
#     "simply_replace_missing_line(each_line,indent)"
#     ]
mutation_operators = ["simply_replace_missing_line(each_line,indent)"]

subject_file_path = {
    "add_specific":"c2",
    "simply_replace":"c1",}

class PerformMutation:
    def __init__(self, subject, input_mode, func_name=None):
        self.input_mode = input_mode
        self.func_name = func_name
        self.subject = subject

    def ast_class_root_mutation(self, CUT, target_file_without_extension, code_length):
        mask_blocks = []
        CUT_lines = CUT.splitlines()
        tree = ast.parse(CUT)
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, ast.ClassDef):
                mask_blocks.append([child.lineno + 1, child.end_lineno])
                self.ast_function_root_mutation(CUT_lines, child, target_file_without_extension, code_length, 1)
        for mask_block in mask_blocks:
            masked_codelines = ast_mutation(CUT_lines, mask_block[0], mask_block[1])
            masked_code = "\n".join(masked_codelines)
            with open(
                    f"{root}/input/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{mask_block[0]}_{mask_block[1]}_0.txt",
                    "w", encoding="utf-8") as f:
                f.write(masked_code)

    def ast_function_root_mutation(self, CUT_lines, tree, target_file_without_extension, code_length, depth):
        mask_blocks = []
        if self.input_mode == "function":
            for child in ast.iter_child_nodes(tree):
                if (isinstance(child, ast.FunctionDef) and child.name == self.func_name
                        and not has_repetition(target_file_without_extension, child.lineno + 1, child.end_lineno)):
                    mask_blocks.append([child.lineno + 1, child.end_lineno])
        elif self.input_mode == "class":
            for child in ast.iter_child_nodes(tree):
                if (isinstance(child, ast.FunctionDef)
                        and not has_repetition(target_file_without_extension, child.lineno + 1, child.end_lineno)):
                    mask_blocks.append([child.lineno + 1, child.end_lineno])
        for mask_block in mask_blocks:
            masked_code = "\n".join(ast_mutation(CUT_lines, mask_block[0], mask_block[1]))
            with open(
                    f"{root}/input/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{mask_block[0]}_{mask_block[1]}_{depth}.txt",
                    "w", encoding="utf-8") as f:
                f.write(masked_code)

    def perform_ast_function_root_mutation(self, CUT, target_file_without_extension, code_length):
        CUT_lines = CUT.splitlines()
        tree = ast.parse(CUT)
        self.ast_function_root_mutation(CUT_lines, tree, target_file_without_extension, code_length, 0)

    def perform_single_mutation(self, target_file:str, output_directory=f"{root}/input", mode="single",
                                merged_line_to_mask=None):
        print("Prepare to perform single line mutation...")
        with open(get_target_code_path(self.subject), 'r', encoding="utf-8") as file1:
            original_code = file1.read()
        target_file_without_extension = target_file.replace(".txt", "")
        original_code_lines = original_code.splitlines()
        comment_lines = []
        is_comment_lines_ended = False
        for line_index, each_line in enumerate(original_code_lines):
            # print(f"line_index:{line_index}")
            lineno = line_index + 1
            if self.input_mode == "function" and not is_comment_lines_ended:
                if "\"\"\"" in each_line:
                    if comment_lines:
                        is_comment_lines_ended = True
                    comment_lines.append(line_index)
                    continue
                elif comment_lines:
                    comment_lines.append(line_index)
                    continue
            if mode == "ast":
                if has_repetition(target_file_without_extension, lineno, lineno):
                    continue
                code_length = len(original_code.splitlines())
                input_file_name = f"{output_directory}/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{lineno}_{lineno}_s.txt"
            if each_line.isspace() or each_line == "" or each_line.strip().startswith("#"): #not "\n" because the lines have already been splitted
                continue
            # Currently do not mutate lines of function/class def etc.
            if any([ word in each_line for word in keywords_cannot_any_mutation]):
                continue
            if self.input_mode == "function" or self.input_mode == "class":
                is_in_mask_block = False
                for depth, mask_depth in merged_line_to_mask.items():
                    if mask_depth:
                        for mask_block in mask_depth:
                            if lineno >= mask_block[0] and line_index <= mask_block[1]:
                                is_in_mask_block = True
                                break
                if not is_in_mask_block:
                    continue
            # print(f"!!!mutation_operators: {mutation_operators}")
            for each_mutation_operator in mutation_operators:
                # print(f"!!!each_mutation_operator: {each_mutation_operator}")
                indent = get_indent(each_line)

                #for specific mutation, we apply various mutation operators to produce mutated code
                option_list = ["simple"]
                if "add_specific" in each_mutation_operator:
                    option_list = define_specific_mutation(indent)
                # print(f"***option_list:{option_list}")
                for idx, each_option in enumerate(option_list):
                    mutated_line,file_label = eval(each_mutation_operator)
                    if mode == "single":
                        input_file_name = f"{output_directory}/{subject_file_path[file_label]}/{file_label}/{target_file_without_extension}/{target_file_without_extension}_{line_index + 1}_{file_label}_{idx + 1}.txt"
                        print(f"!!!input_file_name:{input_file_name}")
                    if os.path.isfile(input_file_name):
                        continue
                    #add_specific() does not perform mutation
                    if mutated_line == "":
                        print("continue")
                        continue
                    code = original_code
                    replaced_code = code.replace(each_line,mutated_line)
                    with open(input_file_name, "w", encoding="utf-8") as file_to_store:
                        file_to_store.write(replaced_code)


def has_repetition(target_file_without_extension, lineno, end_lineno):
    input_files = list_txt_files(f"{root}/input/c1/ast/{target_file_without_extension}")
    for each_output_file in input_files:
        if int(each_output_file.split("_")[-3]) == lineno and int(each_output_file.split("_")[-2]) == end_lineno:
            return True
    return False


def ast_mutation(original_code_lines, starting_line, end_line, enable_template=0) -> list[str]:
    print("Prepare to perform ast mutation...")
    all_lines = []
    indent = ""
    has_masked = False
    for line_num, each_line in enumerate(original_code_lines):
        if line_num + 1 < int(starting_line) or line_num + 1 > int(end_line):
            all_lines.append(each_line)
        elif not has_masked and each_line != "":
            indent = get_indent(each_line)
            # indent = get_indent_for_block(original_code,starting_line,end_line)
            if enable_template == 0:
                all_lines.append(indent + "<MASK>")
            elif enable_template == 1:
                all_lines.append(indent + "if <MASK>:")
                all_lines.append(indent + "    <MASK>")
            elif enable_template == 2:
                all_lines.append(indent + "if <MASK>:")
                all_lines.append(indent + "    <MASK>")
                all_lines.append(indent + "else:")
                all_lines.append(indent + "    <MASK>")
            has_masked = True
    return all_lines


def perform_ast_mutation(target_file:str, starting_line:str,end_line:str,depth:str, enable_template=0, output_directory = f"{root}"):
    target_file_without_extension = target_file.replace(".txt", "")
    with open(get_target_code_path(target_file_without_extension), 'r', encoding="utf-8") as file1:
        original_code = file1.read()
    original_code_lines = original_code.splitlines()
    code_length = len(original_code_lines)
    if os.path.isfile(
            f"{root}/input/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{starting_line}_{end_line}_{depth}.txt"):
        return
    all_lines = ast_mutation(original_code_lines, starting_line, end_line, enable_template)
    with open(f"{output_directory}/input/c1/ast/{target_file_without_extension}/{target_file_without_extension}_{code_length}_ast_{starting_line}_{end_line}_{depth}.txt", "w", encoding="utf-8") as file_to_store:
        for each_line in all_lines:
            print(f"each_line: {each_line}")
            file_to_store.write(each_line + "\n")
        
        # indent = get_indent(each_line)


def main(target:str,model:str, func_name, mode="single", input_mode="console"):
    file_check(target)
    initialisation(target,model)
    target_file_without_extension = target.replace(".txt", "")
    pm = PerformMutation(target_file_without_extension, input_mode, func_name)
    am = AstMutator(input_mode, func_name)
    with open(get_target_code_path(target_file_without_extension), encoding="utf-8") as f:
        target_code = f.read()
    if mode == "single":
        merged_line_to_mask = am.load_and_run(target_code)
        pm.perform_single_mutation(target,merged_line_to_mask=merged_line_to_mask)
    elif mode == "ast":
        merged_line_to_mask = am.load_and_run(target_code)
        for depth in merged_line_to_mask:
            print(f"depth: {depth}, value: {merged_line_to_mask[depth]}")
            for each_pair in merged_line_to_mask[depth]:
                perform_ast_mutation(target, each_pair[0],each_pair[1],depth)
        code_length = len(target_code.splitlines())
        is_skip_depth_0 = False
        if input_mode == "function" or input_mode == "class":
            if input_mode == "function":
                pm.perform_ast_function_root_mutation(target_code, target_file_without_extension, code_length)
            else:
                pm.ast_class_root_mutation(target_code, target_file_without_extension, code_length)
        for block in merged_line_to_mask[1]:
            if block[0] == 1 and block[1] == code_length:
                is_skip_depth_0 = True
        if input_mode == "console" and not is_skip_depth_0:
            perform_ast_mutation(target, "1", str(len(target_code.splitlines())), "0")
        pm.perform_single_mutation(target, mode="ast", merged_line_to_mask=merged_line_to_mask)
        # starting_line
        # end_line
        # depth
        # perform_block_mutation(target,model,starting_line,end_line,depth,output_directory = "subjects")
        # dummy = True
    else:
        raise ValueError(f"Invalid mode: expected 'single' or 'ast', but found {mode}")

if __name__ == '__main__':
    # target = "886d.txt"
    target=sys.argv[1]
    model=sys.argv[2]
    mode=sys.argv[3]  # single or ast
    input_mode = sys.argv[4]  # function or console
    if len(sys.argv) > 5:
        func_name = sys.argv[5]
    else:
        func_name = None
    main(target,model, func_name, mode, input_mode)