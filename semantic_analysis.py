import ast
from multiprocessing import Pool

from func_timeout import FunctionTimedOut, func_set_timeout

import mutation.ast_mutator
from mutation.generate_code_with_llm import GenerateCodeWithLLM
import mutation.perform_mutation as pm
from mutation.utils import list_txt_files, root, is_same_syntax_by_ast, initialisation, file_check, wrap_as_function, \
    analyze_indentation, get_function_name, coarse_grain, class_execute, get_target_code_path
import coverage as covlib
import csv
import importlib
import os
import random
import re
import signal
import sys
import time
from collections import Counter


class Semantic_Analysis:
    DIFF_OUT = 0
    SAME_SYNTAX = 2
    environment = "linux"
    model = "gemini"
    subject: str


    class BlockPair:
        locations = [[], []]
        C1 = []
        C2 = []

        def __init__(self):
            self.locations = [[], []]
            self.C1 = []
            self.C2 = []

    def __init__(self, mode, subject, input_mode, model, environment):
        self.environment = environment
        self.mode = mode
        self.model = model
        self.subject = subject
        self.input_mode = input_mode
        if self.input_mode == "function":
            self.input_mode_class = FunctionMode(self)
        elif self.input_mode == "class":
            self.input_mode_class = ClassMode(self)

    def compare_ignore_space(self, code_list1, code_list2):
        code1 = "\n".join(code_list1)
        code2 = "\n".join(code_list2)
        code1_nospace = code1.replace(" ", "")
        code2_nospace = code2.replace(" ", "")
        return code1_nospace == code2_nospace

    def compare_variable_rename(self, c1_filename, c2_filename, c1_diff, c2_diff, func_name=""):
        if self.input_mode == "console":
            self.adjust_indent(c1_diff)
            self.adjust_indent(c2_diff)
            new_code_lines_c1 = ["def verify():"] + c1_diff
            new_code_lines_c2 = ["def verify():"] + c2_diff
            result = "module1.verify.__code__.co_code == module2.verify.__code__.co_code"
        else:
            new_code_lines_c1 = c1_diff
            new_code_lines_c2 = c2_diff
            result = f"module1.{func_name}.__code__.co_code == module2.{func_name}.__code__.co_code"
        if not os.path.isfile(f"stub_folder/C1_rename_{c1_filename}.py"):
            with open(f"stub_folder/C1_rename_{c1_filename}.py", "w", encoding="utf-8") as f:
                f.write("\n".join(new_code_lines_c1))
        if not os.path.isfile(f"stub_folder/C2_rename_{c2_filename}.py"):
            with open(f"stub_folder/C2_rename_{c2_filename}.py", "w", encoding="utf-8") as f:
                f.write("\n".join(new_code_lines_c2))
        module1 = importlib.import_module(f"stub_folder.C1_rename_{c1_filename}")
        module2 = importlib.import_module(f"stub_folder.C2_rename_{c2_filename}")
        try:
            return eval(result)
        except Exception:
            print("Catch an exception in compare_variable_rename")


    def stub_type(self, c1_filename, c2_filename, first, second, indent_c1, indent_c2):
        """
        :return: [c1_stub_path, c2_stub_path, is_first_empty, is_second_empty]
        """
        first_lines = first.splitlines()
        second_lines = second.splitlines()

        # if len(result.locations[0]) != 0:
        #     c1variables = extract_variables(first, result.locations[0][0], result.locations[0][-1])
        # if len(result.locations[1]) != 0:
        #     c2variables = extract_variables(second, result.locations[1][0], result.locations[1][-1])

        c1_stub_path = f"stub_folder/C1_stub_{c1_filename}.py"
        c2_stub_path = f"stub_folder/C2_stub_{c2_filename}.py"

        if not os.path.isfile(c1_stub_path):
            if self.input_mode == "console":
                c1_code = "\n".join(stub_type_operation(first_lines, indent_c1))
            else:
                c1_code = first
            with open(c1_stub_path, "w", encoding="utf-8") as file:
                file.write(c1_code)
        if not os.path.isfile(c2_stub_path):
            if self.input_mode == "console":
                c2_code = "\n".join(stub_type_operation(second_lines, indent_c2))
            else:
                c2_code = second
            with open(c2_stub_path, "w", encoding="utf-8") as file:
                file.write(c2_code)

        return [c1_stub_path, c2_stub_path]


    def continue_loop(self, first_lines, second_lines, blockpair):
        extra_diff = self.continue_and_break_expand(first_lines, blockpair.locations[0])
        if extra_diff == []:
            extra_diff = self.continue_and_break_expand(second_lines, blockpair.locations[1])
            if extra_diff != []:
                blockpair.locations[0] = self.add_extra_diff(first_lines, blockpair.locations[0], blockpair.locations[1])
        else:
            # add_extra_diff(c1, c2, blockpair.locations[1], extra_diff)
            self.continue_and_break_expand(second_lines, blockpair.locations[1])
        blockpair.C1 = self.reset_diff_code(first_lines, blockpair.locations[0])
        blockpair.C2 = self.reset_diff_code(second_lines, blockpair.locations[1])


    def continue_and_break_expand(self, code, diff_range):
        extra_diff = []
        for lineno in diff_range:
            if "continue" in code[lineno - 1] or "break" in code[lineno - 1]:
                continue_indent = self.calculate_indent(code[lineno - 1])
                for prev_lineno in range(lineno - 1, 0, -1):
                    if "for" in code[prev_lineno - 1] and prev_lineno not in diff_range and self.calculate_indent(
                            code[prev_lineno - 1]) < continue_indent:
                        diff_range.append(prev_lineno)
                        extra_diff.append(prev_lineno)
                        for_indent = self.calculate_indent(code[prev_lineno - 1])
                        for next_lineno in range(prev_lineno + 1, len(code) + 1):
                            if self.calculate_indent(code[next_lineno - 1]) > for_indent:
                                if next_lineno not in diff_range:
                                    diff_range.append(next_lineno)
                                    extra_diff.append(next_lineno)
                            else:
                                break
                        diff_range.sort()
                        return extra_diff
                    if self.calculate_indent(code[prev_lineno - 1]) < continue_indent:
                        continue_indent = self.calculate_indent(code[prev_lineno - 1])
        return extra_diff


    def add_extra_diff(self, modified_code, diff_range, ref_diff_range):
        modified_block_info = mutation.ast_mutator.load_and_run("\n".join(modified_code))
        min_distance = 999
        target_block = diff_range
        for k, v in modified_block_info.items():
            for block in v:
                distance = abs(block[0] - ref_diff_range[0]) + abs(block[-1] - ref_diff_range[-1])
                if distance < min_distance:
                    min_distance = distance
                    target_block = block
        diff_range = target_block
        while True:
            if diff_range[-1] - diff_range[-2] != 1:
                diff_range.insert(-1, diff_range[-2] + 1)
            else:
                return diff_range
        # for lineno in extra_diff:
        #     if ref_code[lineno - 1] in modified_code:
        #         arr = np.array(modified_code)
        #         indices = np.where(arr == ref_code[lineno - 1])
        #         if len(indices[0]) == 1:
        #             result = modified_code.index(ref_code[lineno - 1]) + 1
        #         else:
        #             result = min(indices[0], key=lambda x: abs(x - lineno)) + 1
        #         if result not in diff_range:
        #             diff_range.append(result)
        # diff_range.sort()


    def reset_diff_code(self, code, diff_range):
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


    def pass_each_mutated_input(self, c1_stub_path, c2_stub_path, func_name, input_params, is_reload):
        '''
        :return: out1, err1, out2, err2
        '''
        # out_table1 = dict()
        # out_table2 = dict()
        out_table1 = []
        out_table2 = []
        err1 = False
        err2 = False
        c1_module_name = c1_stub_path.replace(".py", "").replace("/", ".")
        c2_module_name = c2_stub_path.replace(".py", "").replace("/", ".")
        if is_reload:
            c1_module = importlib.reload(importlib.import_module(c1_module_name))
            c2_module = importlib.reload(importlib.import_module(c2_module_name))
        else:
            c1_module = importlib.import_module(c1_module_name)
            c2_module = importlib.import_module(c2_module_name)
        if self.input_mode == "console":
            if self.environment == "linux":
                try:
                    out_table1 = c1_module.quest(input_params, out_table1)
                except FunctionTimedOut:
                    err1 = True
                except Exception:
                    err1 = True
                try:
                    out_table2 = c2_module.quest(input_params, out_table2)
                except FunctionTimedOut:
                    err2 = True
                except Exception:
                    err2 = True
            else:
                try:
                    out_table1 = c1_module.quest(input_params, out_table1)
                except Exception:
                    err1 = True
                try:
                    out_table2 = c2_module.quest(input_params, out_table2)
                except Exception:
                    err2 = True
        else:
            # print(f"input_params:{input_params}")
            # for each_input in input_params:
            #     print(f"input:{each_input}")
            #     try:
            #         out_table1 = eval(f"c1_module.{func_name}({each_input})")
            #     except Exception as e1:
            #         print(f"e1:{e1}")
            #         err1 = False
            #     try:
            #         out_table2 = eval(f"c2_module.{func_name}({each_input})")
            #     except Exception as e2:
            #         print(f"e2:{e2}")
            #         err2 = False
            #     assert False, "debugging"
                        #     print(f"input:{each_input}")
            # print(f"input_params:{input_params}")
            try:
                out_table1 = eval(f"c1_module.{func_name}({input_params})")
            except Exception as e1:
                # print(f"e1:{e1}")
                err1 = False
            try:
                out_table2 = eval(f"c2_module.{func_name}({input_params})")
            except Exception as e2:
                # print(f"e2:{e2}")
                err2 = False
            # assert False, "debugging"
        return out_table1, err1, out_table2, err2


    def pass_each_mutated_input_to_multiple_code(self, code_path_list:list[str]|list[list[list]], func_name, input_params, is_reload):
        """
        :return: [[code 1 result], [code 2 result], ...], [is_code1_exception, is_code2_exception, ...]
        """
        # out_table1 = dict()
        # out_table2 = dict()
        out_table_list = []
        err_list = []
        if func_name != None:
            code_path_list_len = len(code_path_list)
            module_name_path_list = []
            module_list = []
            for code_path_idx in range(code_path_list_len):
                # code_path = code_path_list[code_path_idx].replace(".py", "").replace("/", ".")
                # print(f'!!!code_path:{".".join(code_path_list[code_path_idx-1].split("/")[:-1])}')
                # exec(f'import {".".join(code_path_list[code_path_idx-1].split("/")[:-1])}')
                # print(f'c{code_path_idx}_module_name = {code_path_list[code_path_idx-1]}.replace(".py", "").replace("/", ".")')
                module_name_path_list.append(code_path_list[code_path_idx].replace(".py", "").replace("/", "."))
                # exec(f'c{code_path_idx}_module_name = \"{code_path_list[code_path_idx-1].replace(".py", "").replace("/", ".")}\"')
            # c1_module_name = c1_stub_path.replace(".py", "").replace("/", ".")
            # c2_module_name = c2_stub_path.replace(".py", "").replace("/", ".")
            if is_reload:
                for code_path_idx in range(code_path_list_len):
                    module_list.append(importlib.reload(importlib.import_module(module_name_path_list[code_path_idx])))
                # c1_module = importlib.reload(importlib.import_module(c1_module_name))
                # c2_module = importlib.reload(importlib.import_module(c2_module_name))
            else:
                for code_path_idx in range(code_path_list_len):
                    module_list.append(importlib.import_module(module_name_path_list[code_path_idx]))
                # c1_module = importlib.import_module(c1_module_name)
                # c2_module = importlib.import_module(c2_module_name)


            # if self.input_mode == "console":
            #     if self.environment == "linux":
            #         signal.signal(signal.SIGALRM, self.handler)
            #         for code_path_idx in range(code_path_list_len):
            #             signal.alarm(10)
            #             try:
            #                 out_table_list.append(module_list[code_path_idx].quest(input_params, out_table{code_path_idx}))
            #             except Exception:
            #                 err_list.append(True)
            #             signal.alarm(0)
            #     else:
            #         for code_path_idx in range(code_path_list_len):
            #             signal.alarm(10)
            #             try:
            #                 out_table_list.append(module_list[code_path_idx].quest(input_params, out_table{code_path_idx}))
            #             except Exception:
            #                 err_list.append(True)
            #             signal.alarm(0)
            # else:
            # print(f"!!!module_list:{module_list}")
            for code_path_idx in range(code_path_list_len):
                # print(f"exercising path:{code_path_idx}")
                try:
                    self.function_execution(input_params, code_path_idx, func_name, out_table_list, err_list)
                except FunctionTimedOut:
                    err_list.append(True)
                except Exception as e:
                    # print(f"!!!Exception:{e}")
                    err_list.append(True)
        else:
            for code_path in code_path_list:
                with open(code_path, "r") as code_file:
                    code = code_file.read()
                try:
                    result, err = class_execute(code, input_params)
                except FunctionTimedOut:
                    result = None
                    err = True
                out_table_list.append(result)
                if err:
                    err_list.append(True)
        return out_table_list, err_list

    @func_set_timeout(120)
    def function_execution(self, input_params, code_path_idx, func_name, out_table_list, err_list):
        try:
            # print(f"!!!input_params:{input_params}")
            # print(f"!!!Command:module_list[{code_path_idx}].{func_name}{input_params}")
            if "(" in input_params[0] and ")" in input_params[-1]:
                input_params = input_params[1:-1]
            # output = eval(f"module_list[{code_path_idx}].{func_name}({input_params})")
            output = eval(f"module_list[{code_path_idx}].{func_name}({input_params})")
            if output == None:
                out_table_list.append("Debug-Empty")
            else:
                out_table_list.append(str(output))
        except FunctionTimedOut:
            err_list.append(True)
        except Exception as e:
            # print(f"!!!Exception:{e}")
            err_list.append(True)
    # def complete_execution(first, second, c1_filename, c2_filename):
    #     c1_address = "manual_analysis/" + c1_filename
    #     c2_address = "manual_analysis/" + c2_filename
    #     with open(c1_address, 'w', encoding="utf-8") as f:
    #         f.write(first)
    #     with open(c2_address, 'w', encoding="utf-8") as f:
    #         f.write(second)
    #
    #     c1 = subprocess.Popen(['python', 'C1.py'],
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.PIPE,
    #                           stdin=subprocess.PIPE)
    #     c2 = subprocess.Popen(['python', 'C2.py'],
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.PIPE,
    #                           stdin=subprocess.PIPE)
    #     with open('input.txt', 'r', encoding="utf-8") as f:
    #         input_params = f.read()
    #     out1, err1 = c1.communicate(input=input_params.encode())
    #     out2, err2 = c2.communicate(input=input_params.encode())
    #     err1_str = err1.decode('utf-8')
    #     err2_str = err1.decode('utf-8')
    #     if err1_str != '' or err2_str != '':
    #         return "Exception in original code"
    #     if out1 != out2:
    #         return 3
    #     else:
    #         return 1


    def prune_diff(self, code_lines):
        min_indent = 99
        for i in range(len(code_lines)):
            if code_lines[i].isspace() or code_lines[i] == "":
                continue
            code_lines[i] = code_lines[i].removeprefix("- ")
            code_lines[i] = code_lines[i].removeprefix("+ ")
            current_indent = self.calculate_indent(code_lines[i])
            if current_indent < min_indent:
                min_indent = current_indent
        for i in range(len(code_lines)):
            code_lines[i] = code_lines[i][min_indent:]


    def extract_type(no):
        if no == 1:
            with open('C1_type.txt', encoding="utf-8") as f:
                type_text = f.read()
        else:
            with open('C2_type.txt', encoding="utf-8") as f:
                type_text = f.read()
        type_text = type_text.splitlines()
        table = dict()

        if type_text == "":
            return table

        filtered_table = dict()  # key = 变量名，value = 类型，筛选掉表达式
        for line in type_text:
            # if "class" in line:
            #     pattern = r'{(.*?)}'
            #     type_info = re.findall(pattern, line)[0]
            #     line_list = type_info.split(" ")
            #     table[line_list[0]] = line_list[2].removeprefix("'").removesuffix("'>")
            if "note" in line:
                pattern = r'"(.*?)"'
                type_info = re.findall(pattern, line)[0]
                message = line.split(":")
                with open(message[0], "r", encoding="utf-8") as f:
                    code = f.read()
                code = code.splitlines()
                text = code[int(message[1]) - 1]
                element_pattern = r'\((.*?)\)'
                element = re.findall(element_pattern, text)[0]
                table[element] = type_info
        for k, v in table.items():
            if v != "builtin_function_or_method" and v != "type":
                filtered_table[k] = v
        return filtered_table


    def extract_result(self, output):
        type_text = output.splitlines()
        table = dict()

        if type_text == "":
            return table

        for line in type_text:
            if "{" in line:
                pattern = r'{(.*?)}'
                type_info = re.findall(pattern, line)[0]
                line_list = type_info.split(" ")
                table[line_list[0]] = line_list[1]
        return table
        
    def selecting_test_that_increases_coverage(self,inconsistency_revealing_test_input,stub_type_result,cut_path,func_name,identifier,previous_selected_test):
        cov = covlib.Coverage(data_file=identifier,branch=True)
        final_test_input_list = []
        coverage = 0
        #=======First, compute the coverage achieved by previously selected test===========
        print(f"number of previous tests:{len(previous_selected_test)}")
        cov.start()
        #For computing the coverage of previous test
        if previous_selected_test != []:
            for each_test_cases_candidate in previous_selected_test:
                result_list, exception_list = self.pass_each_mutated_input_to_multiple_code(
                    stub_type_result, func_name, each_test_cases_candidate, False
                )     
                #Compute cut's output
                cut_result_list, cut_exception_list = self.pass_each_mutated_input_to_multiple_code(
                    [cut_path], func_name, each_test_cases_candidate, False
                )
            # cov.stop()
            with open('coverage_report.tmp', 'w') as f:
                coverage = cov.report(file=f,omit="semantic_analysis.py")
        #================================================================================        

        #=======Second, compute the coverage achieved by new test===========
        # cov.start()
        # time.sleep(1)
        # print(f"There are {len(inconsistency_revealing_test_input)} candidates.")
        #Note that there could be more than 10k failing test being found
        if len(inconsistency_revealing_test_input) > 100:
            inconsistency_revealing_test_input = inconsistency_revealing_test_input[:100]
        for idx, each_test_cases_candidate in enumerate(inconsistency_revealing_test_input):
            # print(f"Executing {idx}th inconsistency revealing test.")
            result_list, exception_list = self.pass_each_mutated_input_to_multiple_code(
                stub_type_result, func_name, each_test_cases_candidate, False
            )     
            #Compute cut's output
            cut_result_list, cut_exception_list = self.pass_each_mutated_input_to_multiple_code(
                [cut_path], func_name, each_test_cases_candidate, False
            )
            # time.sleep(1)
            # print(f"each_test_cases_candidate:{each_test_cases_candidate}.")
            with open('coverage_report.tmp', 'w') as f:
                # new_coverage = cov.report(omit="semantic_analysis.py",show_missing=True)
                new_coverage = cov.report(file=f,omit="semantic_analysis.py")
            # new_coverage = 1
            if new_coverage > coverage:
                print(f"The new coverage is:{new_coverage}")
                final_test_input_list.append(each_test_cases_candidate)
                coverage = new_coverage
        #====================================================================================
        print(f"Finish recording coverage")
        cov.stop()
        print(f"Finally selected {len(final_test_input_list)} additional candidates for appending to inconsistency_revealing_test_input.")
        #Note that this function only returns additional test instead of together with previous test, because the append is performed in generate_incosistency_inducing_input() in end2end.py
        return final_test_input_list
            

    def avoid_adding_redundant_test(self,inconsistency_revealing_test_input,resultant_argument_list_arg):

        #Note that no need to add resultant_argument_list to final_test, because in end2end.py the statement is `resultant_argument_list += sa_o.execute_tests(reference_version_path_list, target, target_signature_dict[target],is_diff_testing=False,end2end=True,cut_path=cut_path,resultant_argument_list_arg=resultant_argument_list)`
        final_test = []
        for each_new_test_input in inconsistency_revealing_test_input:
            if each_new_test_input in resultant_argument_list_arg:
                final_test.append(each_new_test_input)
        return final_test



    def execute_tests(self, stub_type_result, identifier, func_name, is_diff_testing=False,end2end=False,cut_path=None,resultant_argument_list_arg=None):
        """
        This function first performs mutation on generated test input  `test_cases += self.mutate_input(each_input)`, then run the test input to see whether the test outputs are consistent/failure-revealing (`outputs_with_max_count != cut_result_list[0]`)

        :param stub_type_result: [c1_stub_path, c2_stub_path], to find failure-revealing test, it is test input pathgenerated 
               resultant_argument_list_arg: previous test, execute_tests() is iteratively called by generate_incosistency_inducing_input in end2end.py
        :return: if is_diff_testing=False, return code that represents if 2 codes are different/identical syntax/semantic, else return failing_inducing_e.g.r.v(the list could be empty)
                Note that the this function has two return statement, one is a normal return, another one is return when timeout, the returned variable is the same (i.e., test cases from generated test/mutated test that can increase coverage)

        code:

        0 diff out of range

        1 same semantic

        3 different semantic

        4 type extraction failed(obsolescent)

        5 testing timeout
        """
        if is_diff_testing or end2end:
            #Store all inconsistency revealing test inputs
            inconsistency_revealing_test_input = []
        test_input_file_list = list_txt_files(f"{root}/test_input/{self.subject}/{self.model}")
        start_time = time.time()
        cov = covlib.Coverage(data_file=identifier)
        while True:  # keeps mutating input, until timeout
            for test_idx, each_input_file in enumerate(test_input_file_list,1):
                with open(each_input_file, encoding="utf-8") as f:
                    each_input = f.read()
                try:
                    test_cases = self.input_mode_class.split_test_cases(each_input)
                except:
                    continue
                for i in range(1000):
                    test_cases.append(self.mutate_input(each_input))
                #This test input candidate is with respect to test input of test_idx
                #=====This loop adds test input to list if it can lead to test input that leads to consistent output========
                test_input_that_leads_to_consistent_output = []
                for each_generated_and_mutated_input in test_cases:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 120:
                        print(f"{identifier} - timeout")
                        if len(inconsistency_revealing_test_input) > 0:
                            #Selecting only test cases that can increase coverage (i.e., removing redundant test cases)
                            final_test_case_list = self.avoid_adding_redundant_test(inconsistency_revealing_test_input,resultant_argument_list_arg)
                            return final_test_case_list
                        else:
                            return []

                    #Only comparing two code
                    invalid_input_output = None
                    result_list, exception_list = self.pass_each_mutated_input_to_multiple_code(
                        stub_type_result, func_name, each_generated_and_mutated_input, False
                    )
                    invalid_input_output = len(exception_list) >= 5  # If a majority of the output is an exception, then we don't need to consider that output

                    if invalid_input_output:
                        continue
                    result_str_list = [str(i) for i in result_list]
                    ct = Counter(result_str_list)
                    if not ct:  # No output has been produced, must be something wrong
                        continue
                    max_count = max(ct.values())

                    outputs_with_max_count = sorted(key for key, value in ct.items() if key and value == max_count)

                    # Only an input that allows majority of reference versions to have consistent output will be considered
                    # Note that unlike the previous branch, this branch does not handle exception, because comparison with CUT is not here but in a following branch
                    if max_count >= 5:
                        # Avoid adding redundant tests
                        if each_generated_and_mutated_input not in resultant_argument_list_arg and each_generated_and_mutated_input not in test_input_that_leads_to_consistent_output:
                            test_input_that_leads_to_consistent_output.append(each_generated_and_mutated_input)
                if test_input_that_leads_to_consistent_output:  # For elements in test_input_that_leads_to_consistent_output, we consider whether the elements are failure-revealing
                    cov.start()
                    for each_test_cases_candidate in test_input_that_leads_to_consistent_output:
                        result_list, exception_list = self.pass_each_mutated_input_to_multiple_code(
                            stub_type_result, func_name, each_test_cases_candidate, False
                        )
                        cut_result_list, cut_exception_list = self.pass_each_mutated_input_to_multiple_code(
                            [cut_path], func_name, each_test_cases_candidate, False
                        )  # Compute cut's output
                        if not cut_result_list or cut_result_list[0] is None:
                            print(f"Debug: {cut_path}")
                            print(f"each_test_cases_candidate: {each_test_cases_candidate}")
                            continue
                        if isinstance(self.input_mode_class, FunctionMode):
                            if isinstance(result_list[0], dict):
                                result_str_list = [str(i) for i in result_list]
                                ct = Counter(result_str_list)
                            else:
                                ct = Counter(result_list)
                            if not ct:  # No output has been produced, must be something wrong
                                continue
                            max_count = max(ct.values())

                            for key, value in ct.items():
                                if value == max_count:
                                    outputs_with_max_count = key
                                    break
                            if isinstance(result_list[0], dict):
                                for result in result_list:
                                    if str(result) == outputs_with_max_count:
                                        outputs_with_max_count = result
                            if outputs_with_max_count != cut_result_list[0]:
                                inconsistency_revealing_test_input.append(each_test_cases_candidate)
                        elif isinstance(self.input_mode_class, ClassMode):
                            for cut_result in cut_result_list[0]:
                                idx = cut_result[0]
                                rv_result_list = []
                                for file_result in result_list:
                                    if file_result is not None:
                                        rv_result_list.append(file_result[idx][1])
                                rv_str_result_list = []
                                for rv_result in rv_result_list:
                                    if isinstance(rv_result, str):
                                        rv_str_result_list.append(f"'{rv_result}'")
                                    else:
                                        rv_str_result_list.append(str(rv_result))
                                ct = Counter(rv_str_result_list)
                                if not ct:
                                    continue
                                max_count = max(ct.values())
                                for key, value in ct.items():
                                    if value == max_count:
                                        outputs_with_max_count = key
                                        break
                                for result in rv_result_list:
                                    if isinstance(result, str):
                                        if f"'{result}'" == outputs_with_max_count:
                                            outputs_with_max_count = result
                                            break
                                    else:
                                        if str(result) == outputs_with_max_count:
                                            outputs_with_max_count = result
                                            break
                                if outputs_with_max_count != cut_result[1]:
                                    inconsistency_revealing_test_input.append(each_test_cases_candidate)
                                    break
                    cov.stop()
                    report = None
                    with open('coverage_report.txt', 'w') as f:
                        print(f"Writing coverage report")
                        report = cov.report(file=f, omit="semantic_analysis.py")
                    with open('coverage_value.tmp', 'w') as f:
                        f.write(str(report))
                    final_test_case_list = self.avoid_adding_redundant_test(inconsistency_revealing_test_input,resultant_argument_list_arg) 
                    return final_test_case_list

    def generate_test_input(self):
        test_input_file_list = list_txt_files(f"{root}/test_input/{self.subject}/{self.model}")
        for each_input_file in test_input_file_list:
            with open(each_input_file, encoding="utf-8") as f:
                each_input = f.read()
            test_cases = [each_input]
            for i in range(1000):
                test_cases.append(self.mutate_input(each_input))
        return test_cases

    def mutate_console_input(self, test_input: str) -> str:
        input_list = test_input.splitlines()
        mutated_input_list = []
        for index, input_line in enumerate(input_list):
            input_line_split = input_line.split()  # e.g."2 3 4 5"
            for i, word in enumerate(input_line_split):
                if isinstance(word, list) or isinstance(word, tuple) or isinstance(word, set):
                    input_line_split[i] = self.mutate_word(word)
                elif str(word).isdigit() or str(word).isnumeric() or isinstance(word, float):
                    input_line_split[i] = self.mutate_number(str(word))
                elif isinstance(word, bool) or word == "True" or word == "False":
                    input_line_split[i] = self.mutate_bool_of_input()
                elif word is None or word == "None":
                    input_line_split[i] = "None"
                elif isinstance(word, dict):
                    input_line_split[i] = self.mutate_dict(word)
                else:
                    input_line_split[i] = self.mutate_str_of_input(word)
            mutated_input_list.append(" ".join(input_line_split))
        return "\n".join(mutated_input_list)

    def mutate_input(self, test_input: str, in_one_line=False, only_one_arg=False) -> list[str]:
        """
        :param in_one_line: All test inputs are written in one line instead of one test input per line.
        :param only_one_arg: One test input comprises only one argument.
        """
        return self.input_mode_class.mutate_input(test_input, in_one_line, only_one_arg)

    def mutate_word(self, word):
        if isinstance(word, list):
            if not word:
                return []
            operation = random.choice(["remove", "repeat", "insert", "replace"])
            index = random.randint(0, len(word) - 1)
            if operation == "remove":
                word.pop(index)
            elif operation == "repeat":
                word.insert(index + 1, word[index])
            elif operation == "insert":
                insert_index = random.randint(0, len(word))
                word.insert(insert_index, self.mutate_word(word[index]))
            else:
                xi = word.pop(index)
                word.insert(index, self.mutate_word(xi))
            return word
        elif isinstance(word, tuple):
            return tuple(self.mutate_word(list(word)))
        elif isinstance(word, set):
            return set(self.mutate_word(list(word)))
        elif isinstance(word, bool):
            return self.mutate_bool_of_input()
        elif isinstance(word, int) or isinstance(word, float):
            return self.mutate_number(word)
        elif word is None:
            return None
        elif isinstance(word, dict):
            return self.mutate_dict(word)
        else:
            return self.mutate_str_of_input(word)

    def mutate_number(self, word):
        increment = random.choice([-1, 1])
        return word + increment

    def mutate_bool_of_input(self):
        return random.choice([True, False])

    def mutate_dict(self, input):
        if input == {}:
            return {}
        operation = random.choice(['remove', 'update', 'insert'])
        key_to_operate = random.choice(list(input.keys()))
        if operation == 'remove':
            del input[key_to_operate]
        elif operation == 'update':
            input[key_to_operate] = self.mutate_word(input[key_to_operate])
        else:
            input[self.mutate_word(key_to_operate)] = self.mutate_word(input[key_to_operate])
        return input

    def mutate_str_of_input(self, input):
        if len(input) > 1:
            sub_len = random.randint(1, len(input))
            start = random.randint(0, len(input) - sub_len)
        else:
            sub_len = 1
            start = 0
        substr = input[start:start + sub_len]
        operation = random.choice(['repeat', 'remove', 'substr'])
        if operation == 'repeat':
            return input.replace(substr, substr + substr)
        elif operation == 'remove':
            return input.replace(substr, '')
        else:
            return input.replace(substr, self.mutate_str_of_input(substr))


    def compare_list(self, list1, list2):
        t = list(list2)
        try:
            for elem in list1:
                t.remove(elem)
        except ValueError:
            return False
        return not t


    # def generate_input(type, test_scale):
    #     test_case = []
    #     if type == "builtins.int":
    #         for i in range(test_scale):
    #             test_case.append(random.randint(0, 100))
    #     elif type == "builtins.list[builtins.int]":
    #         for i in range(test_scale):
    #             length = random.randint(1, 10)
    #             my_list = []
    #             for i in range(length):
    #                 my_list.append(random.randint(0, 100))
    #                 test_case.append(copy.deepcopy(my_list))
    #     elif type == "builtins.list[builtins.list[builtins.int]]":
    #         for i in range(test_scale):
    #             length = random.randint(1, 10)
    #             result = []
    #             for j in range(length):
    #                 length2 = random.randint(1, 10)
    #                 my_list = []
    #                 for i in range(length2):
    #                     my_list.append(random.randint(0, 100))
    #                 result.append(copy.deepcopy(my_list))
    #             test_case.append(copy.deepcopy(result))
    #     elif type == "builtins.str":
    #         for i in range(test_scale):
    #             test_case.append(random_string(26))
    #     elif type == "collections.deque":
    #         for i in range(test_scale):
    #             test_case.append(deque())
    #     return copy.deepcopy(test_case)


    # def random_string(length):
    #     # 生成一个由大写字母、小写字母和数字组成的字符串
    #     characters = string.ascii_letters + string.digits
    #     return ''.join(random.choice(characters) for _ in range(length))


    def adjust_indent(self, test_code):
        """
        This function can adjust non-standard indent to 4-space indent. Note that the minimum indent of result is 4.
        """
        min_indent = 99
        for code_line in test_code:
            if self.calculate_indent(code_line) < min_indent:
                min_indent = self.calculate_indent(code_line)
        adjust_num = 4 - min_indent
        if adjust_num > 0:
            extra_indent = ' ' * adjust_num
            for i, item in enumerate(test_code):
                test_code[i] = extra_indent + test_code[i]
        elif adjust_num < 0:
            for i, item in enumerate(test_code):
                test_code[i] = test_code[i][-adjust_num:]


    def fix_test(self, file_name):
        with open(file_name, encoding='utf-8') as file:
            code = file.read()
        code_lines = code.splitlines()
        for i in range(len(code_lines)):
            if 'continue' in code_lines[i]:
                code_lines[i] = re.sub("continue", "return diction", code_lines[i])
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write("\n".join(code_lines))


    # def differential_testing(mask_location):
    #     with open('./C1.py', encoding="utf-8") as f1:
    #         first = f1.read()
    #     f1.close()
    #     with open('./C2.py', encoding="utf-8") as f2:
    #         second = f2.read()
    #     f2.close()
    #
    #     result = diff(mask_location, first, second)
    #     if result == 0:
    #         print("Syntax of C1 and C2 are same.")
    #         return
    #
    #     with open('./C1snippet.txt', "w") as f3:
    #         min_indent1 = 99
    #         for i in range(len(result.C1)):
    #             result.C1[i] = result.C1[i].removeprefix("- ")
    #             result.C1[i] = result.C1[i].removeprefix("+ ")
    #             result.C1[i] = result.C1[i] + "\n"
    #             indent_snippet1 = len(result.C1[i]) - len(result.C1[i].lstrip())
    #             if indent_snippet1 < min_indent1:
    #                 min_indent1 = indent_snippet1
    #         for code in result.C1:
    #             code = code[min_indent1:]
    #             f3.write(code)
    #
    #     with open('./C2snippet.txt', "w") as f4:
    #         min_indent2 = 99
    #         for i in range(len(result.C2)):
    #             result.C2[i] = result.C2[i].removeprefix("- ")
    #             result.C2[i] = result.C2[i].removeprefix("+ ")
    #             result.C2[i] = result.C2[i] + "\n"
    #             indent_snippet2 = len(result.C2[i]) - len(result.C2[i].lstrip())
    #             if indent_snippet2 < min_indent2:
    #                 min_indent2 = indent_snippet2
    #         for code in result.C2:
    #             code = code[min_indent2:]
    #             f4.write(code)
    #     f4.close()
    #
    #     expand_diff(first, result.locations[0])
    #     expand_diff(second, result.locations[1])
    #
    #     c1variables = extract_variables(first, result.locations[0][0], result.locations[0][-1])  # 从diff中提取相关变量
    #     c2variables = extract_variables(second, result.locations[1][0], result.locations[1][-1])
    #
    #     last_diff_code1 = result.C1[-1]
    #     if ":" in last_diff_code1:
    #         extra_indent = 4
    #     else:
    #         extra_indent = 0
    #     indent1 = ' '.join(['' for _ in range(indent(last_diff_code1) + 1 + extra_indent)])
    #     for variable in c1variables:
    #         insert_text_at_line("./C1.py", result.locations[0][-1],
    #                             indent1 + "print(type(" + variable + "))\n")
    #         # insert_text_at_line("./C1.py", result.locations[0][-1],
    #         #                     indent1 + "print(\"" + variable + "=\"," + variable + ")\n")
    #
    #     last_diff_code2 = result.C2[-1]
    #     if ":" in last_diff_code2:
    #         extra_indent = 4
    #     else:
    #         extra_indent = 0
    #     indent2 = ' '.join(['' for _ in range(indent(last_diff_code2) + 1 + extra_indent)])
    #     for variable in c2variables:
    #         insert_text_at_line("./C2.py", result.locations[1][-1],
    #                             indent2 + "print(type(" + variable + "))\n")
    #         # insert_text_at_line("./C2.py", result.locations[1][-1],
    #         #                     indent2 + "print(\"" + variable + "=\"," + variable + ")\n")
    #
    #     with open("./C1.py", "r") as f7:
    #         first = f7.readlines()
    #     with open("./C2.py", "r") as f8:
    #         second = f8.readlines()
    #     for line in result.locations[0]:
    #         first[line - 1] = first[line - 1].replace("print(", "print(\"diff:\", ")
    #     for line in result.locations[1]:
    #         second[line - 1] = second[line - 1].replace("print(", "print(\"diff:\", ")
    #     with open("./C1.py", "w") as f:
    #         for code in first:
    #             f.write(code)
    #     with open("./C2.py", "w") as f:
    #         for code in second:
    #             f.write(code)
    #
    #     while True:
    #         # generate_input()
    #         c1 = subprocess.Popen(['python', 'C1.py'],
    #                               stdout=subprocess.PIPE,
    #                               stdin=subprocess.PIPE)
    #         c2 = subprocess.Popen(['python', 'C2.py'],
    #                               stdout=subprocess.PIPE,
    #                               stdin=subprocess.PIPE)
    #         with open('input.txt', 'r', encoding="utf-8") as f:
    #             input_params = f.read()
    #         f.close()
    #         out1, err1 = c1.communicate(input=input_params.encode())
    #         out2, err2 = c2.communicate(input=input_params.encode())
    #         if err1 is not None and err2 is not None:
    #             if err1 == err2:
    #                 a = 1
    #                 # generate_input()
    #             else:
    #                 break
    #         else:
    #             break
    #     with open('C1_debug.txt', 'w', encoding="utf-8") as f:
    #         f.write(out1.decode('utf-8'))
    #     with open('C2_debug.txt', 'w', encoding="utf-8") as f:
    #         f.write(out2.decode('utf-8'))
    #
    #     with open('C1_debug.txt', 'r', encoding="utf-8") as f:
    #         debug1 = f.readlines()
    #     print_diff1 = []
    #     diff_vars1 = []
    #     current_diff1 = copy.deepcopy(Assignment())
    #     for line in debug1:
    #         if line.startswith("diff"):  # 收集print()的差异
    #             print_diff1.append(line)
    #         elif "=" in line:
    #             current_diff1.left, current_diff1.right = line.split("= ")
    #             diff_vars1.append(copy.deepcopy(current_diff1))
    #
    #     with open('C2_debug.txt', 'r', encoding="utf-8") as f:
    #         debug2 = f.readlines()
    #     print_diff2 = []
    #     diff_vars2 = []
    #     current_diff2 = copy.deepcopy(Assignment())
    #     for line in debug2:
    #         if line.startswith("diff"):  # 收集print()的差异
    #             print_diff2.append(line)
    #         elif "=" in line:
    #             current_diff2.left, current_diff2.right = line.split("= ")
    #             diff_vars2.append(copy.deepcopy(current_diff2))
    #     if print_diff1 != print_diff2:
    #         print("C1 and C2 print different contents, so they are semantically different.")
    #     elif diff_vars1 != diff_vars2:
    #         print("Variables in C1 and C2 are different, so they are semantically different.")
    #     else:
    #         print("C1 and C2 are semantically the same.")


    # def generate_input():
    #     with open('./prompt.txt', encoding="utf-8") as fp:
    #         words = fp.read()
    #     fp.close()
    #     message = [{"role": "user", "content": words}]
    #     client = OpenAI(api_key=apikey.apiKey)
    #     completion = client.chat.completions.create(
    #         model="gpt-3.5-turbo-0613",
    #         messages=message
    #     )
    #     with open('input.txt', 'w', encoding="utf-8") as f:
    #         f.write(completion.choices[0].message.content)


    def calculate_indent(self, code):
        """
        The function is used to check indent of single line code.
        """
        return len(code) - len(code.lstrip())


    def expand_diff(self, code_lines, diff_range):
        extra_diff = []
        if len(diff_range) == 0:
            return []

        first_diff = diff_range[0] - 1
        first_diff_code = code_lines[first_diff]

        # Handle the case where the preceding code in a diff is more indented than the following code
        first_line_indent = self.calculate_indent(first_diff_code)
        for lineno in diff_range:
            if self.calculate_indent(code_lines[lineno - 1]) < first_line_indent:
                for prev_lineno in range(first_diff, 1, -1):
                    diff_range.insert(0, prev_lineno)
                    extra_diff.append(prev_lineno)
                    if self.calculate_indent(code_lines[prev_lineno - 1]) == self.calculate_indent(code_lines[lineno - 1]):
                        break
                break

        first_diff = diff_range[0] - 1
        first_diff_code = code_lines[first_diff]

        if "elif" in first_diff_code or "else" in first_diff_code:
            for lineno in range(first_diff, 1, -1):
                diff_range.insert(0, lineno)
                extra_diff.append(lineno)
                if "if" in code_lines[lineno - 1] and "elif" not in code_lines[lineno - 1]:
                    break
            if not first_diff + 1 in diff_range:
                for i in range(first_diff + 1, len(code_lines)):
                    if self.calculate_indent(code_lines[i]) > self.calculate_indent(first_diff_code):
                        diff_range.append(i + 1)
                        extra_diff.append(i + 1)
                    else:
                        break

        last_diff = diff_range[-1] - 1
        last_diff_code = code_lines[last_diff]
        last_diff_code_strip = last_diff_code.strip()
        if len(last_diff_code_strip) != 0:
            if last_diff_code_strip[-1] == ":" or self.is_if_statement(last_diff_code):
                for i in range(1, len(code_lines) - last_diff):
                    if self.calculate_indent(code_lines[last_diff + i]) > self.calculate_indent(last_diff_code):
                        diff_range.append(last_diff + i + 1)
                        extra_diff.append(last_diff + i + 1)
                    else:
                        break

        if "return" in code_lines[diff_range[-1] - 1]:
            delete_line = diff_range[-1]
            del diff_range[-1]
            if delete_line in extra_diff:
                del extra_diff[extra_diff.index(delete_line)]
        return extra_diff


    def is_if_statement(self, code_line):
        pattern = r'\s*if\s+[^:]+(?:\s*:\s*.*)?$'
        return re.match(pattern, code_line) is not None


    def extract_variables(self, code, start_line, end_line):
        # 解析代码为AST
        parsed_code = ast.parse(code)

        # 用于存储提取的变量
        variables = []

        # 遍历AST节点
        for node in ast.walk(parsed_code):
            # 获取节点所在的行号
            if hasattr(node, 'lineno'):
                line_number = node.lineno
                # 如果在指定范围内
                if line_number > end_line:
                    # 提取赋值的变量名
                    if isinstance(node, ast.Name) and not self.is_excluded_node(node, ast.walk(parsed_code)):
                        variables.append(node.id)

        return variables


    def is_excluded_node(self, target_node, ast_iterator):
        for node in ast_iterator:
            if isinstance(node, ast.Call):
                if target_node == node.func:
                    return True
            if isinstance(node, ast.For):
                if target_node == node.target:
                    return True
        return False


    def find_assignments(self, node, variables):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    for name in target.elts:
                        variables.append(name.id)
                else:
                    variables.append(target.id)
        for child in ast.iter_child_nodes(node):
            self.find_assignments(child, variables)
        return variables


    def insert_text_at_line(self, file_path, line_number, text_to_insert):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines.insert(line_number, text_to_insert)

        with open(file_path, 'w') as file:
            for line in lines:
                file.write(line)


    # def is_same_syntax_by_string(code1, code2):
    #     return distance(code1, code2) < 4


    # def is_same_syntax_by_ast(tree1, tree2):
    #     return distance(ast.dump(tree1), ast.dump(tree2)) < 1
        # if type(tree1) != type(tree2):
        #     return False
        # 比较节点字段，这里假设字段都是公共的
        # for field in tree1._fields:
        #     value1 = getattr(tree1, field, None)
        #     value2 = getattr(tree2, field, None)
        #
        #     if isinstance(value1, ast.AST):
        #         if not is_same_syntax_by_ast(value1, value2):
        #             return False
        #     elif isinstance(value1, list):
        #         if len(value1) != len(value2):
        #             return False
        #         for item1, item2 in zip(value1, value2):
        #             if isinstance(item1, ast.AST):
        #                 if not is_same_syntax_by_ast(item1, item2):
        #                     return False
        #             elif item1 != item2:
        #                 return False
        #     elif value1 != value2:
        #         return False


    def executing_file(self, c1_path, c2_path, is_diff_testing):
        """
        :return: if is_diff_testing=False, return code that represents if 2 codes are different/identical syntax/semantic, else return failing_inducing_e.g.r.v(the list could be empty)

        code:

        -1 SyntaxError in code

        0 diff out of range

        1 same semantic

        2 same syntax

        3 different semantic

        4 type extraction failed(obsolescent)

        5 testing timeout
        """
        with open(c1_path, encoding="utf-8") as f:
            first = f.read()
        with open(c2_path, encoding="utf-8") as f:
            second = f.read()
        c1_filename = c1_path.split("/")[-1].removesuffix(".txt")
        c2_filename = c2_path.split("/")[-1].removesuffix(".txt")
        identifier = f"{c1_filename}_{c2_filename.split('_')[-1]}"

        indent1 = analyze_indentation(first)
        indent2 = analyze_indentation(second)

        first = self.remove_comments(first)
        second = self.remove_comments(second)

        str1_lines = first.splitlines()
        str2_lines = second.splitlines()
        for i in range(len(str1_lines) - 1, -1, -1):
            if not str1_lines[i].strip():
                del str1_lines[i]
        for i in range(len(str2_lines) - 1, -1, -1):
            if not str2_lines[i].strip():
                del str2_lines[i]
        first = "\n".join(str1_lines)
        second = "\n".join(str2_lines)

        try:
            tree1 = ast.parse(first)
            func_name = get_function_name(first) if self.input_mode == "function" else ""
            tree2 = ast.parse(second)
        except SyntaxError:
            print(f"{identifier} - SyntaxError in code")
            return -1
        if check_has_no_output(str1_lines) or check_has_no_output(str2_lines):
            print(f"{identifier} - No output in code")
            return -1
        if is_same_syntax_by_ast(tree1, tree2):
            print(f"{identifier} - Same syntax.")
            return 2
        else:
            if self.compare_variable_rename(c1_filename, c2_filename, str1_lines, str2_lines, func_name):
                print(f"{identifier} - Same syntax.(with variable renamed)")
                return self.SAME_SYNTAX
            else:
                stub_type_result = self.stub_type(c1_filename, c2_filename, first, second, indent1, indent2)
                return self.execute_tests(stub_type_result, identifier, func_name, is_diff_testing)


    def remove_comments(self, source):
        # 删除字符串开头的井号（#）直到行尾的所有内容
        clean_source = re.sub(r'#.*?\n|#+[ ]*[\r\n]', '\n', source)
        clean_source = re.sub(r'(?s)"""(?:[^"\\]|\\.)*?"""', '\n', clean_source)
        return clean_source


    # def traverse_file(input_folder, c1_folder, c2_folder):
    #     data = []
    #     for root, dirs, files in os.walk(input_folder):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             subject = os.path.splitext(file)[0]
    #             subject_index = len(data)
    #             data.append([subject])
    #             with open(file_path, encoding="utf-8") as f:
    #                 input = f.read()
    #             with open('input.txt', "w", encoding="utf-8") as f:
    #                 f.write(input)
    #             for c1_root, c1_dirs, c1_files in os.walk(c1_folder):
    #                 for c1_file_name in c1_files:
    #                     c1_subject = c1_file_name.split("_")[0]
    #                     c1_lineno = c1_file_name.split("_")[1]
    #                     if c1_subject == subject:
    #                         c1_no = c1_file_name.split("_")[-1]
    #                         c1_index = len(data)
    #                         data.append([c1_lineno + "-" + c1_no])
    #                         c1_path = os.path.join(c1_root, c1_file_name)
    #                         with open(c1_path, encoding="utf-8") as f:
    #                             c1_content = f.read()
    #                         with open('C1.py', "w", encoding="utf-8") as f:
    #                             f.write(c1_content)
    #                         for c2_root, c2_dirs, c2_files in os.walk(c2_folder):
    #                             for c2_file_name in c2_files:
    #                                 c2_subject = c2_file_name.split("_")[0]
    #                                 c2_lineno = c2_file_name.split("_")[1]
    #                                 if c2_subject == subject and c2_lineno == c1_lineno:
    #                                     c2_type = c2_file_name.split("_")[-2]
    #                                     c2_no = c2_file_name.split("_")[-1]
    #                                     if c2_type + "-" + c2_no not in data[subject_index]:
    #                                         c2_index = len(data[subject_index])
    #                                         data[subject_index].append(c2_type + "-" + c2_no)
    #                                     else:
    #                                         c2_index = data[subject_index].index(c2_type + "-" + c2_no)
    #                                     c2_path = os.path.join(c2_root, c2_file_name)
    #                                     with open(c2_path, encoding="utf-8") as f:
    #                                         c2_content = f.read()
    #                                     with open('C1.py', "w", encoding="utf-8") as f:
    #                                         f.write(c1_content)
    #                                     with open('C2.py', "w", encoding="utf-8") as f:
    #                                         f.write(c2_content)
    #                                     try:
    #                                         print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no, " start")
    #                                         result_code = executing_file(int(c1_lineno), c1_file_name)
    #                                         if result_code == 0:
    #                                             insert_result(data[c1_index], c2_index, "diff out of range")
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                         elif result_code == 1:
    #                                             insert_result(data[c1_index], c2_index, "same semantic")
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                         elif result_code == 2:
    #                                             insert_result(data[c1_index], c2_index, "same syntax")
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                         elif result_code == 3:
    #                                             insert_result(data[c1_index], c2_index, "different semantic")
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                         elif result_code == 4:
    #                                             insert_result(data[c1_index], c2_index, "type extraction failed")
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                         else:
    #                                             insert_result(data[c1_index], c2_index, result_code)
    #                                             print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                                   " done")
    #                                     except TimeoutError:
    #                                         insert_result(data[c1_index], c2_index, "timeout")
    #                                         print(subject, "-", c1_lineno, "-", c1_no, "-", c2_type, "-", c2_no,
    #                                               " timeout")
    #                                         error_log.write(f"TimeoutError:{subject}-{c1_lineno}-{c1_no}-{c2_type}-{c2_no}")
    #                                         continue
    #                                         # with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #                                         #     writer = csv.writer(csvfile)
    #                                         #     for row in data:
    #                                         #         writer.writerow(row)
    #                                         # return
    #                                     except RecursionError:
    #                                         error_log.write(
    #                                             f"RecursionError:{subject}-{c1_lineno}-{c1_no}-{c2_type}-{c2_no}")
    #                                         continue
    #     error_log.close()
    #     with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for row in data:
    #             writer.writerow(row)


    # This function is only for inserting csv file.
    def insert_result(self, list, index, result):
        if index == len(list):
            list.append(result)
        elif index < len(list):
            list[index] = result
        else:
            list.append("")
            self.insert_result(list, index, result)


    # def execute_with_timeout(func, subject, mask_location, timeout):
    #     def target():
    #         nonlocal result
    #         result = func(subject, mask_location)
    #
    #     result = None
    #     thread = threading.Thread(target=target)
    #     thread.daemon = True
    #     thread.start()
    #     thread.join(timeout)
    #     if thread.is_alive():
    #         thread._stop()
    #         raise TimeoutError("Function execution timed out")
    #     return result


    def reset_c1_test_c2_test(self):
        with open('./C1_test.py', "w") as f1:
            f1.write("")
        with open('./C2_test.py', "w") as f1:
            f1.write("")


    def mixed_consistent(self, input_folder, c1_folder, c2_folder):
        data = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                subject = os.path.splitext(file)[0]
                data.append([subject])
                with open(file_path, encoding="utf-8") as f:
                    input = f.read()
                with open('input.txt', "w", encoding="utf-8") as f:
                    f.write(input)
                for c1_root, c1_dirs, c1_files in os.walk(c1_folder):
                    for c1_file_name in c1_files:
                        c1_subject = c1_file_name.split("_")[0]
                        c1_lineno = c1_file_name.split("_")[1]
                        if c1_subject == subject:
                            c1_no = c1_file_name.split("_")[-1]
                            c1_index = len(data)
                            data.append([c1_lineno + "-" + c1_no])
                            same_semantics = []
                            c1_path = os.path.join(c1_root, c1_file_name)
                            with open(c1_path, encoding="utf-8") as f:
                                c1_content = f.read()
                            with open('mixed_compare/C1.py', "w", encoding="utf-8") as f:
                                f.write(c1_content)
                            file_count = 1
                            for c2_root, c2_dirs, c2_files in os.walk(c2_folder):
                                for c2_file_name in c2_files:
                                    c2_subject = c2_file_name.split("_")[0]
                                    c2_lineno = c2_file_name.split("_")[1]
                                    c2_no = c2_file_name.split("_")[-1]
                                    if c2_subject == subject and c2_lineno == c1_lineno and c1_no == c2_no:
                                        c2_type = c2_file_name.split("_")[-2]
                                        c2_path = os.path.join(c2_root, c2_file_name)
                                        with open(c2_path, encoding="utf-8") as f:
                                            c2_content = f.read()
                                        file_count += 1
                                        file_address = "mixed_compare/C" + str(file_count) + ".py"
                                        with open(file_address, "w", encoding="utf-8") as f:
                                            f.write(c2_content)
                            for root_mc, dirs_mc, files_mc in os.walk('mixed_compare'):
                                for filename in files_mc:
                                    current_first = int(filename[-4])
                                    current_second = current_first
                                    first_address = "mixed_compare/C" + str(current_first) + ".py"
                                    with open(first_address, encoding="utf-8") as f:
                                        c1_content = f.read()
                                    with open('C1.py', "w", encoding="utf-8") as f:
                                        f.write(c1_content)
                                    while current_second < file_count:
                                        current_second += 1
                                        second_address = "mixed_compare/C" + str(current_second) + ".py"
                                        with open(second_address, encoding="utf-8") as f:
                                            c2_content = f.read()
                                        with open('C2.py', "w", encoding="utf-8") as f:
                                            f.write(c2_content)
                                        try:
                                            c2_file_name = "c2.txt"
                                            result_code = self.executing_file(int(c1_lineno), c1_file_name)
                                            if result_code == 1:
                                                if len(same_semantics) > 0:
                                                    for i, item in enumerate(same_semantics):
                                                        if current_first in item:
                                                            if current_second not in item:
                                                                same_semantics[i].append(current_second)
                                                        elif current_second in item:
                                                            if current_first not in item:
                                                                same_semantics[i].append(current_first)
                                                        else:
                                                            same_semantics.append([current_first, current_second])
                                                else:
                                                    same_semantics.append([current_first, current_second])
                                        except TimeoutError:
                                            print(subject, "-", c1_lineno, "-", c1_no, "-",
                                                  " timeout")
                            is_accepted = 0
                            if len(same_semantics) != 0:
                                for pair in same_semantics:
                                    if len(pair) >= 3:
                                        is_accepted = 1
                            if is_accepted == 1:
                                self.insert_result(data[c1_index], 1, "accepted")
                            else:
                                self.insert_result(data[c1_index], 1, "rejected")

        with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

    def run(self, sorted_divided_group, CUT_path):
        data = []
        explored_nodes = []
        gcllm = mutation.generate_code_with_llm.GenerateCodeWithLLM(f"{self.subject}.txt", self.model, self.mode,
                                                                    False, self.input_mode)
        has_failure_revealing_test_case = False
        same_semantics = []
        test_inputs = []
        data.append([subject, "result"])
        for index, row in sorted_divided_group.iterrows():
            current_depth = str(row["depth"])
            current_start = int(row["start"])
            current_end = int(row["end"])
            key = f"{current_depth}_{current_start}_{current_end}"
            filepath = row["filepath"]
            skip_child_node = False
            for node in explored_nodes:
                node_start = int(node.split("_")[1])
                node_end = int(node.split("_")[2])
                if current_start >= node_start and current_end <= node_end:
                    skip_child_node = True
                    break
            if skip_child_node:
                continue
            gcllm.perform_infilling_for_one_file(filepath)
            c1_files = list_txt_files(f"{root}/output/c1/{self.mode}/{self.subject}/{self.model}")
            code_group = []
            for c1_file in c1_files:
                c1_depth = c1_file.split("_")[-2]
                c1_start = int(c1_file.split("_")[-4])
                c1_end = int(c1_file.split("_")[-3])
                if current_depth == c1_depth and current_start == c1_start and current_end == c1_end:
                    code_group.append(c1_file)
            execute_file_count = len(code_group)
            ordinate = [x[0] for x in data]
            if key in ordinate:
                vertical_index = ordinate.index(key)
            else:
                vertical_index = len(data)
                data.append([key])
            is_accepted = 0
            code_dict = {}
            for index, py_file in enumerate(code_group):
                code_dict[py_file] = []
                second_index = index
                while second_index < execute_file_count - 1:
                    second_index += 1
                    second_address = code_group[second_index]
                    code_dict[py_file].append(second_address)
            input_list = []
            for cd_key, values in code_dict.items():
                for value in values:
                    input_list.append([cd_key, value])
            egrv_set = set()
            with Pool(processes=os.cpu_count()) as p:
                print(subject, "-", key, "- start")
                result_code_list = [p.apply_async(self.executing_file, (input[0], input[1], False)) for input in input_list]
                for index, res in enumerate(result_code_list):
                    try:
                        result_code = res.get(timeout=180)
                    except Exception:
                        print("Infinite loop!")
                        result_code = 5
                    if result_code == 1:
                        first_address = input_list[index][0]
                        current_first = input_list[index][0].split("_")[-1].removesuffix(".txt")
                        current_second = input_list[index][1].split("_")[-1].removesuffix(".txt")
                        if len(same_semantics) > 0:
                            for i, item in enumerate(same_semantics):
                                if current_first in item:
                                    if current_second not in item:
                                        same_semantics[i].append(current_second)
                                elif current_second in item:
                                    if current_first not in item:
                                        same_semantics[i].append(current_first)
                                else:
                                    same_semantics.append([current_first, current_second])
                        else:
                            same_semantics.append([current_first, current_second])
                        if len(same_semantics) != 0:
                            for pair in same_semantics:
                                if len(pair) >= 2:
                                    is_accepted = 1
                                    self.insert_result(data[vertical_index], 1, "accepted")
                        data[vertical_index] = data[vertical_index] + same_semantics
                        same_semantics = []
                        egrv_set.add(first_address)
                        # explored_nodes.append(key)
            if egrv_set != set():
                egrv_file_path = list(egrv_set)[0]
                result = self.executing_file(CUT_path, egrv_file_path, True)
                if result:
                    if isinstance(result, list):
                        # for index_tc, failure_revealing_test_case in enumerate(result):
                        #     with open(
                        #         f"output/{subject}_ast_{current_depth}_{current_start}_{index_tc}.txt", "w",
                        #         encoding="utf-8"
                        #     ) as f:
                        #         f.write(failure_revealing_test_case)
                        test_inputs = test_inputs + result
                        has_failure_revealing_test_case = True
                if is_accepted == 0:
                    self.insert_result(data[vertical_index], 1, "rejected")
                    same_semantics = []
        if not has_failure_revealing_test_case:
            print("failure-revealing test case cannot be found")
        else:
            cov = covlib.Coverage()
            with open(CUT_path, encoding="utf-8") as f:
                original_code = f.read()
            original_code_indent = analyze_indentation(original_code)
            original_code_lines = original_code.splitlines()
            original_code_lines = wrap_as_function(original_code_lines, original_code_indent, "coverage")
            with open("original_code.py", "w", encoding="utf-8") as file:
                file.write("\n".join(original_code_lines))
            branch_coverage = 0
            valid_test_inputs = []
            for test in test_inputs:
                cov.start()
                module = importlib.import_module("original_code")
                try:
                    module.quest(test)
                except Exception:
                    pass
                cov.stop()
                with open('origin_coverage_report.txt', 'w') as f:
                    report = cov.report(file=f, omit="main.py")
                if report > branch_coverage:
                    valid_test_inputs.append(test)
                    branch_coverage = report
                if report == 100:
                    for index_tc, failure_revealing_test_case in enumerate(valid_test_inputs):
                        with open(
                                f"output/{subject}_ast_{index_tc}.txt", "w",
                                encoding="utf-8"
                        ) as f:
                            f.write("\n".join(failure_revealing_test_case))
                    break
            if branch_coverage != 100:
                print("The branch coverage of the entire code by all test inputs cannot reach 100%.")
                for index_tc, failure_revealing_test_case in enumerate(valid_test_inputs):
                    with open(
                            f"output/{subject}_ast_{index_tc}.txt", "w",
                            encoding="utf-8"
                    ) as f:
                        f.write("\n".join(failure_revealing_test_case))
        print(f"writing report...")
        self.write_egrv_report(f"{subject}_egrv.csv", data)

    def list_py_files(self, directory):
        txt_files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isdir(filepath):
                txt_files += self.list_py_files(filepath)
            elif filename.endswith('.py'):
                txt_files.append(filepath)
        txt_files.sort()
        return txt_files


    def write_egrv_report(self, file_path, content):
        if os.path.isfile(file_path):
            count = 1
            while True:
                if os.path.isfile(file_path.removesuffix(".csv") + f"_{count}.csv"):
                    count += 1
                else:
                    file_path = file_path.removesuffix(".csv") + f"_{count}.csv"
                    break
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in content:
                writer.writerow(row)


    def convert_seconds(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h{minutes}min{seconds}s"


class FunctionMode:
    def __init__(self, super_class: Semantic_Analysis):
        self.super_class = super_class

    def mutate_input(self, test_input: str, in_one_line=False, only_one_arg=False) -> list[str]:
        """
        :param in_one_line: All test inputs are written in one line instead of one test input per line.
        :param only_one_arg: One test input comprises only one argument.
        """
        input_list = test_input.splitlines()
        if in_one_line:
            input_list = list(eval(input_list[0]))  # input_list[0] is all test inputs when in_one_line==True
        mutated_input_list = []
        for index, input_line in enumerate(input_list):
            input_line_split = eval(input_line)  # change str to original type
            mutated_input_line = []
            if only_one_arg:
                mutated_input_line.append(self.super_class.mutate_word(input_line_split))
            else:
                for i, word in enumerate(input_line_split):
                    mutated_input_line.append(self.super_class.mutate_word(word))
            mutated_input_line_str = str(mutated_input_line)
            mutated_input_list.append(mutated_input_line_str.removeprefix("[").removesuffix("]"))
        return mutated_input_list

    def split_test_cases(self, test_cases):
        return [eval(i) for i in test_cases.splitlines()]


class ClassMode:
    def __init__(self, super_class: Semantic_Analysis):
        self.super_class = super_class

    def mutate_input(self, test_input: str, in_one_line=False, only_one_arg=False) -> list[str]:
        input_list = test_input.splitlines()
        mutated_input_list = []
        for each_input in input_list:
            each_input_strip = each_input.rstrip()
            args_str = extract_last_parentheses_content(each_input_strip)
            args = split_function_args(args_str)
            mutated_args = []
            if args:
                for arg in args:
                    try:
                        if detect_str(arg):
                            result = self.super_class.mutate_word(arg[1:-1])
                        else:
                            result = self.super_class.mutate_word(eval(arg))
                        if isinstance(result, str):
                            mutated_args.append(f"\"{result}\"")
                        else:
                            mutated_args.append(str(result))
                    except NameError:
                        mutated_args.append(arg)
            mutated_args_str = ", ".join(mutated_args)
            mutated_input_list.append(each_input[:each_input.find("(") + 1] + mutated_args_str + ")")
        return mutated_input_list

    def split_test_cases(self, test_cases):
        return [test_cases.splitlines()]


def check_has_no_output(code_lines):
    code_lines_without_empty_print = []
    for index, code in enumerate(code_lines):
        if code.strip() != "print()":
            code_lines_without_empty_print.append(code)
    code_without_empty_print = "\n".join(code_lines_without_empty_print)
    if "print(" not in code_without_empty_print and "return" not in code_without_empty_print:
        return True
    else:
        return False


def detect_str(arg: str):
    if arg.startswith('\'') or arg.startswith('\"'):
        return True
    return False


def extract_last_parentheses_content(s):
    # 从字符串的末尾开始寻找最后一个右括号的位置
    right_index = -1
    left_index = -1
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ')':
            right_index = i
            break

    # 如果没有找到右括号，则返回空字符串
    if right_index == -1:
        return ""

    # 继续向左寻找与这个右括号相匹配的左括号
    open_count = 0
    is_inside_string = False
    for i in range(right_index, -1, -1):
        if s[i] == ')':
            if not is_inside_string:
                open_count += 1
        elif s[i] == '(':
            if not is_inside_string:
                open_count -= 1
                if open_count == 0:
                    left_index = i
                    break
        elif s[i] == '"':
            if not is_inside_string:
                is_inside_string = True
            else:
                is_inside_string = False

    # 如果找到了一对匹配的括号，则返回它们之间的字符串
    if left_index != -1 and right_index != -1 and left_index < right_index:
        return s[left_index + 1:right_index]
    else:
        return ""


def identify_args_punctuation(s):
    is_inside_string = False
    bracket_stack = []
    is_inside_bracket = False
    result = []
    for idx, char in enumerate(s):
        if char == ',':
            if not is_inside_string and not is_inside_bracket:
                result.append(idx)
        if char == '"' or char == '\'':
            if not is_inside_string:
                is_inside_string = True
            else:
                is_inside_string = False
        elif char == '{' or char == '(' or char == '[':
            if not is_inside_string:
                is_inside_bracket = True
                bracket_stack.append(char)
        elif char == '}' or char == ')' or char == ']':
            if not is_inside_string:
                bracket_stack.pop()
                if not bracket_stack:
                    is_inside_bracket = False
    return result


def delete_keyword_arguments(s):
    is_inside_string = False
    result = []
    for idx, char in enumerate(s):
        if char == '=':
            if not is_inside_string:
                result.append(idx)
        elif char == '"':
            if not is_inside_string:
                is_inside_string = True
            else:
                is_inside_string = False
    delete_indexs = []
    for equal in result:
        has_delete_index = False
        for i in range(equal, -1, -1):
            if s[i] == "," or s[i] == "(":
                delete_indexs.append([i+1, equal])
                has_delete_index = True
                break
        if not has_delete_index:
            delete_indexs.append([0, equal])
    delete_indexs.reverse()
    for idx_pair in delete_indexs:
        s = s[:idx_pair[0]] + s[idx_pair[1] + 1:]
    return s


def split_function_args(s) -> list[str]:
    if s == "":
        return []
    s_without_keyword_arguments = delete_keyword_arguments(s)
    if not identify_args_punctuation(s):
        return [s_without_keyword_arguments]
    # pattern = r'[^,\s\[]+?\([^\)]*\),\s*|,\s*[^,\s]+?\([^\)]*\)|[^,\s\[]+?\([^\)]*\)'  # 正则表达式匹配自定义函数/类
    # match = re.findall(pattern, s)
    # s_without_function = s
    # match_without_punc = []
    # for function in match:
    #     s_without_function = s_without_function.replace(function, '')
    #     match_without_punc.append(function.strip().removeprefix(',').removesuffix(','))
    # if s_without_function == "":
    #     return [s]
    # eval_tuple = ast.literal_eval(s_without_function)
    # if isinstance(eval_tuple, tuple):
    #     evals = list(eval_tuple)
    # else:
    #     evals = [eval_tuple]
    # for idx, ele in enumerate(evals):
    #     if isinstance(ele, str):
    #         evals[idx] = f"\"{ele}\""
    #     else:
    #         evals[idx] = str(ele)
    # result = match_without_punc + evals
    # return result
    punctuation = identify_args_punctuation(s_without_keyword_arguments)
    result = []
    former_idx = 0
    if not punctuation:
        return [s_without_keyword_arguments]
    for each_punc in punctuation:
        result.append(s_without_keyword_arguments[former_idx:each_punc].strip())
        former_idx = each_punc + 1
    result.append(s_without_keyword_arguments[former_idx:].strip())
    return result


def stub_type_operation(code_lines, code_indent):
    # is_empty = False
    # if len(result_code) == 0:
    #     is_empty = True
    # else:
    #     variables = list(set(variables))
    #     last_diff_code = result_code[-1]
    #     if ":" in last_diff_code[-1]:
    #         extra_indent = 4
    #     elif len(code_lines) == result_location[-1]:
    #         extra_indent = -4
    #     else:
    #         extra_indent = 0
    #     indent = ' '.join(['' for _ in range(calculate_indent(last_diff_code) + 1 + extra_indent)])
    #     for variable in variables:
    #         check = indent + "if \'" + variable + "\' in locals():"
    #         text = indent + " " * code_indent + "print(" + variable + ")"
    #         insert_text = [check, text]
    #         code_lines = code_lines[:result_location[-1]] + insert_text + code_lines[result_location[-1]:]

    code_lines = wrap_as_function(code_lines, code_indent, "stub")
    for i, item in enumerate(code_lines):
        code_lines[i] = item.replace("print(*", "print(")
        code_lines[i] = re.sub(r'print\(', r'diction.append(', code_lines[i])
        code_lines[i] = re.sub(r'stdout\.write\(', r'diction.append(', code_lines[i])
    return code_lines


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    is_debug_mode = sys.argv[1]
    subject = sys.argv[2]
    input_mode = sys.argv[3]
    model = sys.argv[4]
    if is_debug_mode == "1":
        mask_location = int(sys.argv[4])
        env_command = sys.argv[5]
        with open("C1_test.py", encoding="utf-8") as f:
            first = f.read()
        with open("C2_test.py", encoding="utf-8") as f:
            second = f.read()
        sa = Semantic_Analysis("debug", subject, input_mode, model, env_command)
        print(sa.executing_file(first, second, False))
    else:
        analysis_mode = sys.argv[5]  # mixed and ast
        env_command = sys.argv[6]
        input_list = f"{root}/test_input/{subject}/{model}"
        start_time = time.time()
        sa = Semantic_Analysis(analysis_mode, subject, input_mode, model, env_command)
        if analysis_mode == "mixed":
            sa.mixed_consistent("./input", "./c1", "./c2")
        elif analysis_mode == "ast":
            target = f"{subject}.txt"
            prompt_mode = "part"
            pm.main(target, model, analysis_mode, input_mode)
            initialisation(target, model, prompt_mode)
            file_check(target)
            gcllm = GenerateCodeWithLLM(target, model, analysis_mode, True, input_mode)
            gcllm.iterate_all_files(target, model)
            sa.run(coarse_grain(subject, "ast"), get_target_code_path(subject))
            print(sa.convert_seconds(time.time() - start_time))
        print("Completed!")
