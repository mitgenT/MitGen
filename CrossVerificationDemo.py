from func_timeout import func_set_timeout, FunctionTimedOut
import os
import sys
from tqdm import tqdm

from mutation import utils
from mutation.utils import list_txt_files_without_subdir, root, Identifier, coarse_grain, cross_validation, \
    create_identifiers, load_args, get_target_code_path

sys.set_int_max_str_digits(10000)


class Statistics:
    def __init__(self):
        self.count = 0
        self.cv = 0
        self.count_filter = 0
        self.total_filter = 0


class TruePositiveStatistics(Statistics):
    def __init__(self):
        super().__init__()


class FalsePositiveStatistics(Statistics):
    def __init__(self):
        super().__init__()


class FalseNegativeStatistics(Statistics):
    def __init__(self):
        super().__init__()


class TrueNegativeStatistics(Statistics):
    def __init__(self):
        super().__init__()


class DoubleKeyStatistics:
    """
    In result list, 0 represents failure-revealing, 1 false positive, 2 pass correct, 3 pass wrong
    """
    def __init__(self, key_1="", key_2="", result_list=None):
        if result_list is None:
            result_list = []
        self.key_1 = key_1
        self.key_2 = key_2
        self.result_list = result_list
        self.discount_rate = 0

    def calculate(self):
        tp_count = 0
        fp_count = 0
        tn_count = 0
        fn_count = 0
        for result in self.result_list:
            if result == 0:
                tp_count += 1
            elif result == 1:
                fp_count += 1
            elif result == 2:
                tn_count += 1
            elif result == 3:
                fn_count += 1
        cal_result = [tp_count, fp_count, tn_count, fn_count]
        for index, count in enumerate(cal_result):
            cal_result[index] *= (2 - self.discount_rate)
        return cal_result


class DiscountStatistics():
    def __init__(self, dks_list):
        self.dks_list: list[DoubleKeyStatistics] = dks_list

    def calculate(self):
        key_list = {}
        for dks in self.dks_list:
            if dks.result_list:
                if dks.key_1 not in key_list:
                    key_list[dks.key_1] = 1
                else:
                    key_list[dks.key_1] += 1
                if dks.key_2 not in key_list:
                    key_list[dks.key_2] = 1
                else:
                    key_list[dks.key_2] += 1
        return key_list

    def discount(self, origin_dks_list=None):
        match origin_dks_list:
            case None:
                key_list = self.calculate()
                for key in list(key_list):
                    value = key_list[key]
                    if value != 1:
                        for dks in self.dks_list:
                            if dks.key_1 == key or dks.key_2 == key:
                                dks.discount_rate += 1-1/value
            case _:
                for dks in self.dks_list:
                    for origin_dks in origin_dks_list:
                        if origin_dks.key_1 == dks.key_1 and origin_dks.key_2 == dks.key_2:
                            dks.discount_rate = origin_dks.discount_rate
                            break


class IOStatistics:
    def __init__(self, file_list: list, result=None):
        self.type_enum = ["Failure-revealing", "False Positive", "Pass Correct", "Pass Wrong", "Not confirmed"]
        self.type = self.type_enum[4]
        self.result = result
        self.file_list = file_list


class StatisticsLine:
    def __init__(self, index=None, content=None):
        self.index = index
        self.content: list[IOStatistics] = content

    def count_most_common(self):
        self.content.sort(key=lambda x: len(x.file_list), reverse=True)


class StatisticsBoard:
    def __init__(self):
        self.content: list[StatisticsLine] = []

    def add_result(self, index, result, file):
        for line in self.content:
            if index == line.index:
                for io in line.content:
                    if result == io.result:
                        io.file_list.append(file)
                        return
                line.content.append(IOStatistics([file], result))
                return
        self.content.append(StatisticsLine(index, [IOStatistics([file], result)]))


class CrossVerificationDemo:
    def __init__(self, execute_class, subject, model, arguments=None):
        self.tp = TruePositiveStatistics()
        self.tn = TrueNegativeStatistics()
        self.fp = FalsePositiveStatistics()
        self.fn = FalseNegativeStatistics()
        self.arguments = arguments
        self.execute_class = execute_class
        self.model = model
        self.subject = subject
        self.threshold = 10
        self.output_filename = check_file(f"{subject}{model}")

    def iterate(self):
        cv_result = cross_validation(coarse_grain(self.subject, "ast"))
        for key, value in cv_result.items():
            max_depth = calculate_max_depth(value)
            for current_depth_int in range(int(key.depth) + 1, max_depth + 1):
                current_depth = str(current_depth_int)
                child_parent_dict = {}
                for child in value:
                    if child.depth == current_depth:
                        child_parent_dict[child] = [key]
                        for parent_depth in range(int(key.depth) + 1, current_depth_int):
                            parent_depth = str(parent_depth)
                            for identifier in value:
                                if identifier.depth == parent_depth and child.start >= identifier.start and child.end <= identifier.end:
                                    child_parent_dict[child].append(identifier)
                        for parent_identifier in child_parent_dict[child]:
                            print(parent_identifier.toString(), child.toString())
                            result = self.baseline(
                                [f"mutation/subjects/output/c1/ast/{self.subject}/{self.model}"], parent_identifier.toString(),
                                child.toString())
            if True:
                current_depth = "s"
                child_parent_dict = {}
                for child in value:
                    if child.depth == current_depth:
                        child_parent_dict[child] = [key]
                        for parent_depth in range(int(key.depth) + 1, max_depth + 1):
                            parent_depth = str(parent_depth)
                            for identifier in value:
                                if identifier.depth == parent_depth and child.start >= identifier.start and child.end <= identifier.end:
                                    child_parent_dict[child].append(identifier)
                        for parent_identifier in child_parent_dict[child]:
                            print(parent_identifier.toString(), child.toString())
                            result = self.baseline(
                                [f"mutation/subjects/output/c1/ast/{self.subject}/{self.model}"], parent_identifier.toString(),
                                child.toString())

    def baseline(self, output_folder: list, identifier_1: str, identifier_2: str):
        result = 0
        result_text = [f"{identifier_1} - {identifier_2}"]
        target_file_path_list = self.get_target_file_path_list(output_folder, identifier_1, identifier_2)
        if not target_file_path_list[0] or not target_file_path_list[1]:
            return
        board, cut_result_list, correct_result_list = self.execute_class.iterate_module(target_file_path_list[2], self.subject, self.arguments)
        for index, correct_result in correct_result_list:
            line = None
            cut_result = None
            for line_temp in board.content:
                if line_temp.index == index:
                    line = line_temp
                    break
            for cut_result_temp in cut_result_list:
                if cut_result_temp[0] == index:
                    cut_result = cut_result_temp[1]
            result_text = self.calculate(identifier_1, identifier_2, line, cut_result, correct_result, result_text)
        result_text.append("")
        with open(f"output/{self.output_filename}.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(result_text))
        return result

    def calculate(self, identifier_1, identifier_2, line: StatisticsLine, cut_result, correct_result, result_text):
        line.count_most_common()
        result_text.append(f"Input-Output Pair {line.index}: ")
        is_most_common = True
        for io in line.content:
            rv_result = io.result
            count = len(io.file_list)
            files = io.file_list
            if rv_result is not None and rv_result != "Exception":
                verify_1 = False
                verify_2 = False
                for file in files:
                    if identifier_1 in file:
                        verify_1 = True
                    elif identifier_2 in file:
                        verify_2 = True
                if rv_result == cut_result:
                    if rv_result == correct_result:
                        type = "pass correct"
                        sta = self.tn
                    else:
                        type = "pass wrong"
                        sta = self.fn
                else:
                    if rv_result == correct_result:
                        type = "Failure revealed!"
                        sta = self.tp
                    else:
                        type = "false positive"
                        sta = self.fp
                if not (verify_1 and verify_2):
                    type += " (not verified)"
                result_text[-1] = result_text[-1] + f"[[{type}, {count}, {rv_result}] - {files}]"
                if is_most_common:
                    sta.count += 1
                    if (verify_1 and verify_2) and (count >= self.threshold):
                        sta.total_filter += 1
                        result = 1
                    if verify_1 and verify_2:
                        sta.cv += 1
                    if count >= self.threshold:
                        sta.count_filter += 1
            is_most_common = False
        print()
        return result_text

    def get_target_file_path_list(self, output_folder: list, identifier_1, identifier_2):
        files = []
        for folder in output_folder:
            files += list_txt_files_without_subdir(folder)
        target_file_path_list = []
        identifier1_has_file = False
        identifier2_has_file = False
        for filename in files:
            filename_split = filename.split('_')
            identifier = Identifier(filename_split[-2], filename_split[-4], filename_split[-3])
            if identifier_1 == identifier.toString():
                identifier1_has_file = True
                target_file_path_list.append(filename)
            elif identifier_2 == identifier.toString():
                identifier2_has_file = True
                target_file_path_list.append(filename)
        return [identifier1_has_file, identifier2_has_file, target_file_path_list]

    def print_info(self):
        type_enum = [["Failure-revealing", self.tp], ["False Positive", self.fp], ["Pass Correct", self.tn],
                     ["Pass Wrong", self.fn]]
        print("---Statistics---")
        for type_name, type_class in type_enum:
            print(f"{type_name}: {type_class.count}")
            if type_class.count > 0:
                print(f"After cross verification: {type_class.cv} ({type_class.cv / type_class.count*100:.2f}%)")
                print(f"After count filter: {type_class.count_filter} ({type_class.count_filter / type_class.count*100:.2f}%)")
                print(f"After total filter: {type_class.total_filter} ({type_class.total_filter / type_class.count*100:.2f}%)")


class CrossVerificationDiscount(CrossVerificationDemo):
    def __init__(self, execute_class, subject, model, arguments=None):
        super().__init__(execute_class, subject, model, arguments)

    def iterate(self, docstring_verification=False):
        dks_list = []
        dks_cross_list = []
        dks_count_list = []
        dks_total_list = []
        cv_result = cross_validation(coarse_grain(self.subject, "ast"))
        output_folders = []
        if os.path.exists(f"mutation/subjects/output/c1/docstring_verifier/{self.subject}/{self.model}/1"):
            for each_idx in range(1, 11):
                output_folders.append(
                    f"mutation/subjects/output/c1/docstring_verifier/{self.subject}/{self.model}/{each_idx}")
        else:
            output_folders = [f"mutation/subjects/output/c1/ast/{self.subject}/{self.model}"]
        for key, value in cv_result.items():
            max_depth = calculate_max_depth(value)
            for current_depth_int in range(int(key.depth) + 1, max_depth + 1):
                current_depth = str(current_depth_int)
                child_parent_dict = {}
                for child in value:
                    if child.depth == current_depth:
                        child_parent_dict[child] = [key]
                        for parent_depth in range(int(key.depth) + 1, current_depth_int):
                            parent_depth = str(parent_depth)
                            for identifier in value:
                                if identifier.depth == parent_depth and child.start >= identifier.start and child.end <= identifier.end:
                                    child_parent_dict[child].append(identifier)
                        for parent_identifier in child_parent_dict[child]:
                            print(parent_identifier.toString(), child.toString())
                            result = self.baseline(output_folders, parent_identifier.toString(), child.toString())
                            dks_list.append(result[0])
                            dks_cross_list.append(result[1])
                            dks_count_list.append(result[2])
                            dks_total_list.append(result[3])
            if True:
                current_depth = "s"
                child_parent_dict = {}
                for child in value:
                    if child.depth == current_depth:
                        child_parent_dict[child] = [key]
                        for parent_depth in range(int(key.depth) + 1, max_depth + 1):
                            parent_depth = str(parent_depth)
                            for identifier in value:
                                if identifier.depth == parent_depth and child.start >= identifier.start and child.end <= identifier.end:
                                    child_parent_dict[child].append(identifier)
                        for parent_identifier in child_parent_dict[child]:
                            print(parent_identifier.toString(), child.toString())
                            result = self.baseline(output_folders, parent_identifier.toString(), child.toString())
                            dks_list.append(result[0])
                            dks_cross_list.append(result[1])
                            dks_count_list.append(result[2])
                            dks_total_list.append(result[3])
        ds = DiscountStatistics(dks_list)
        ds.discount()
        type_enum = [[self.tp, 0], [self.fp, 1], [self.tn, 2], [self.fn, 3]]
        for dks in ds.dks_list:
            cal_result = dks.calculate()
            for type_name, no in type_enum:
                type_name.count += cal_result[no]
        ds_cross = DiscountStatistics(dks_cross_list)
        ds_cross.discount(ds.dks_list)
        for dks in ds_cross.dks_list:
            cal_result = dks.calculate()
            for type_name, no in type_enum:
                type_name.cv += cal_result[no]
        ds_count = DiscountStatistics(dks_count_list)
        ds_count.discount(ds.dks_list)
        for dks in ds_count.dks_list:
            cal_result = dks.calculate()
            pass
            for type_name, no in type_enum:
                type_name.count_filter += cal_result[no]
        ds_total = DiscountStatistics(dks_total_list)
        ds_total.discount(ds.dks_list)
        for dks in ds_total.dks_list:
            cal_result = dks.calculate()
            for type_name, no in type_enum:
                type_name.total_filter += cal_result[no]

    def baseline(self, output_folder, identifier_1: str, identifier_2: str):
        self.discount = Discount(identifier_1, identifier_2, self.execute_class)
        self.calculate = self.discount.calculate
        super().baseline(output_folder, identifier_1, identifier_2)
        return [self.discount.dks, self.discount.dks_cross, self.discount.dks_count, self.discount.dks_total]

    def print_info(self):
        type_enum = [["Failure-revealing", self.tp], ["False Positive", self.fp], ["Pass Correct", self.tn],
                     ["Pass Wrong", self.fn]]
        text_lines = ["---Discount Statistics---"]
        for type_name, type_class in type_enum:
            text_lines.append(f"{type_name}: {type_class.count}")
            if type_class.count > 0:
                text_lines.append(f"After cross verification: {type_class.cv} ({type_class.cv / type_class.count*100:.2f}%)")
                text_lines.append(f"After count filter: {type_class.count_filter} ({type_class.count_filter / type_class.count*100:.2f}%)")
                text_lines.append(f"After total filter: {type_class.total_filter} ({type_class.total_filter / type_class.count*100:.2f}%)")
        text = "\n".join(text_lines)
        print(text)
        with open(f"output/statistics/{self.output_filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)


class Discount:
    def __init__(self, identifier_1, identifier_2, execute_class):
        self.dks = DoubleKeyStatistics(identifier_1, identifier_2)
        self.dks_count = DoubleKeyStatistics(identifier_1, identifier_2)
        self.dks_cross = DoubleKeyStatistics(identifier_1, identifier_2)
        self.dks_total = DoubleKeyStatistics(identifier_1, identifier_2)
        self.threshold = 10
        self.execute_class = execute_class

    def calculate(self, identifier_1, identifier_2, line: StatisticsLine, cut_result, correct_result, result_text):
        if line == None:
            return [f"{identifier_1}, {identifier_2}, Generated code for this line does not exist"]
        line.count_most_common()
        print(f"Input-Output Pair {line.index}: ", end='')
        result_text.append(f"Input-Output Pair {line.index}: ")
        is_most_common = True
        #Iterate each line
        for io in line.content:
            rv_result = io.result
            #count is number of files that have this output
            count = len(io.file_list)
            files = io.file_list
            #The result is not empty or an exception
            if rv_result is not None and rv_result != "Exception":
                verify_1 = False
                verify_2 = False
                verify_1_count = 0
                verify_2_count = 0
                threshold = 50
                #Iterate each file
                for file in files:
                    if identifier_1 in file:
                        verify_1 = True
                        verify_1_count += 1
                    elif identifier_2 in file:
                        verify_2 = True
                        verify_2_count += 1
                try:
                    if rv_result == cut_result:
                        if rv_result == correct_result:
                            type = "pass correct"
                            type_code = 2
                        else:
                            type = "pass wrong"
                            type_code = 3
                    else:
                        if rv_result == correct_result:
                            type = "Failure revealed!"
                            type_code = 0
                        else:
                            type = "false positive"
                            type_code = 1
                except Exception:
                    continue
                if not (verify_1 and verify_2):
                    type += " (not verified)"
                if not ((verify_1_count + verify_2_count) >= self.threshold):
                    type += " (not pass threshold)"
                # result_text[-1] = result_text[-1] + f"[[{type}, {count}, {rv_result}] - {files}]"
                result_text[-1] = result_text[-1] + f"[{type}, {count} - {files}]"
                if is_most_common:
                    self.dks.result_list.append(type_code)
                    if (verify_1 and verify_2) and (count >= self.threshold):
                        self.dks_total.result_list.append(type_code)
                    if verify_1 and verify_2:
                        self.dks_cross.result_list.append(type_code)
                    if count >= self.threshold:
                        self.dks_count.result_list.append(type_code)
            is_most_common = False
        print()
        return result_text

    def execute(self, module, argument):
        return self.execute_class.execute(module, argument)


class DifferentialPrompting:
    def __init__(self, execute_class, subject, model, argument_list = None):
        self.tp = TruePositiveStatistics()
        self.tn = TrueNegativeStatistics()
        self.fp = FalsePositiveStatistics()
        self.fn = FalseNegativeStatistics()
        self.arguments = argument_list
        self.execute_class = execute_class
        self.model = model
        self.subject = subject
        self.output_collections = {}
        self.output_filename = check_file(f"{subject}{model}_dp")

    def iterate(self, docstring_verification=False):
        identifiers = create_identifiers(coarse_grain(self.subject, "ast"))
        for identifier in identifiers:
            print(identifier.toString())
            self.baseline(f"mutation/subjects/output/c1/ast/{self.subject}/{self.model}", identifier.toString(),
                         identifier.toString())

    def baseline(self, output_folder, identifier: str, identifier_2: str):
        files = list_txt_files_without_subdir(output_folder)
        result_text = self.write_identifier(identifier)
        target_file_path_list = self.get_output_file_path_list(files, identifier, identifier_2)
        if not target_file_path_list:
            return
        counter = 1
        has_consistent_result = False
        while target_file_path_list:
            if has_consistent_result:
                break
            print(self.print_identifier(identifier, counter))
            result_text.append(f"No.{counter}")
            file_path_pair = []
            try:
                file_path_pair.append(target_file_path_list.pop(0))
                file_path_pair.append(target_file_path_list.pop(0))
            except IndexError:
                break
            code_list = []
            for index, file in enumerate(file_path_pair):
                with open(file, 'r', encoding="utf-8") as f:
                    code = f.read()
                code_list.append(code)
            board, cut_result_list, correct_result_list = self.execute_class.iterate_module(file_path_pair, self.subject, self.arguments)
            if len(board.content) == 1 and board.content[0].content[0].result == "Exception":
                counter += 1
                continue
            for index, correct_result in correct_result_list:
                line = None
                cut_result = None
                for line_temp in board.content:
                    if line_temp.index == index:
                        line = line_temp
                for cut_result_temp in cut_result_list:
                    if cut_result_temp[0] == index:
                        cut_result = cut_result_temp[1]
                result_text.append(f"Input-Output Pair {index}: ")
                if len(line.content) == 1:
                    output = line.content[0].result
                    if output is not None and output != "Exception":
                        has_consistent_result = True
                        if output == cut_result:
                            if output == correct_result:
                                type = "pass correct"
                                sta = self.tn
                            else:
                                type = "pass wrong"
                                sta = self.fn
                        else:
                            if output == correct_result:
                                type = "Failure revealed!"
                                sta = self.tp
                            else:
                                type = "false positive"
                                sta = self.fp
                        # result_text[-1] = result_text[-1] + f"[[{type}, {output}] - {file_path_pair}]"
                        result_text[-1] = result_text[-1] + f"[[{type}, {file_path_pair}]"
                        sta.count += 1
                else:
                    result_text[-1] = result_text[-1] + f"[[{line.content[0].result} - {file_path_pair[0]}], [{line.content[1].result} - {file_path_pair[1]}]]"
            counter += 1
        print()
        result_text.append("")
        with open(f"output/{self.output_filename}.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(result_text))
        if not has_consistent_result:
            raise NotImplementedError

    def write_identifier(self, identifier: str):
        return [f"{identifier}"]

    def get_output_file_path_list(self, files, identifier, identifier_2):
        target_file_path_list = []
        for filename in files:
            filename_split = filename.split('_')
            file_identifier = Identifier(filename_split[-2], filename_split[-4], filename_split[-3])
            if identifier == file_identifier.toString() or identifier_2 == file_identifier.toString():
                target_file_path_list.append(filename)
        return target_file_path_list

    def print_identifier(self, identifier: str, counter):
        return f"{identifier}: No.{counter}"

    def print_info(self):
        type_enum = [["Failure-revealing", self.tp], ["False Positive", self.fp], ["Pass Correct", self.tn],
                     ["Pass Wrong", self.fn]]
        text_lines = ["---Differential Prompting Statistics---"]
        for type_name, type_class in type_enum:
            text_lines.append(f"{type_name}: {type_class.count}")
        text = "\n".join(text_lines)
        print(text)
        with open(f"output/statistics/{self.output_filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)


class SingleDepth:
    def __init__(self, execute_class, subject, model, argument_list = None):
        self.tp = TruePositiveStatistics()
        self.tn = TrueNegativeStatistics()
        self.fp = FalsePositiveStatistics()
        self.fn = FalseNegativeStatistics()
        self.execute_class = execute_class
        self.model = model
        self.output_filename = None
        if argument_list == None:
            self.arguments = argument_list
            self.output_filename = check_file(f"{subject}{model}_sd")
        else:
            self.arguments = self.format_argument_list(argument_list)
            self.output_filename = check_file(f"{subject}{model}_sd_g")
        self.subject = subject
        self.threshold = 5

    @staticmethod
    def format_argument_list(arguments):
        argument_list = eval(arguments)
        result = []
        for arg in argument_list:
            result.append(str(arg))
        return str(result)

    def iterate(self, docstring_verification=False):
        identifiers = create_identifiers(coarse_grain(self.subject, "ast"))
        files = []
        #=====docstring_verification=====
        if docstring_verification == True:
            docstring_verification_path_list = []
            for each_idx in range(1, 11):
                docstring_verification_path_list.append(
                    f"mutation/subjects/output/c1/docstring_verifier/{self.subject}/{self.model}/{each_idx}")
            for each_path in docstring_verification_path_list:
                files += list_txt_files_without_subdir(each_path)
        #=====docstring_verification=====
        if not files:
            files = list_txt_files_without_subdir(f"mutation/subjects/output/c1/ast/{self.subject}/{self.model}")
        if not files:
            raise FileNotFoundError(f"Warning: output files not found in mutation/subjects/output/c1/ast/{self.subject}/{self.model}")
        for identifier in identifiers:
            self.baseline(files, identifier.toString(),
                         identifier.toString())

    #argument_list is the input
    def baseline(self, files, identifier_1: str, identifier_2: str):
        print(f"{identifier_1}")
        target_file_path_list = []
        result = 0
        result_text = [f"{identifier_1}"]
        for filename in files:
            filename_split = filename.split('_')
            identifier = Identifier(filename_split[-2], filename_split[-4], filename_split[-3])
            if identifier_1 == identifier.toString():
                target_file_path_list.append(filename)
        if not target_file_path_list:
            return
        board, cut_result_list, correct_result_list = self.execute_class.iterate_module(target_file_path_list, self.subject, self.arguments)
        for index, correct_result in correct_result_list:
            line = None
            cut_result = None
            for line_temp in board.content:
                if line_temp.index == index:
                    line = line_temp
            for cut_result_temp in cut_result_list:
                if cut_result_temp[0] == index:
                    cut_result = cut_result_temp[1]
            result_text.append(f"Input-Output Pair {index}: ")
            is_most_common = True
            for io in line.content:
                rv_result = io.result
                count = len(io.file_list)
                files = io.file_list
                if rv_result is not None and rv_result != "Exception":
                    try:
                        if rv_result == cut_result:
                            if rv_result == correct_result:
                                type = "pass correct"
                                sta = self.tn
                            else:
                                type = "pass wrong"
                                sta = self.fn
                        else:
                            if rv_result == correct_result:
                                type = "Failure revealed!"
                                sta = self.tp
                            else:
                                type = "false positive"
                                sta = self.fp
                    except Exception:
                        continue
                    if count < self.threshold:
                        type += " (not verified)"
                    # result_text[-1] = result_text[-1] + f"[[{type}, {count}, {rv_result}] - {files}]"
                    result_text[-1] = result_text[-1] + f"[[{type}, {count} - {files}]"
                    if is_most_common:
                        sta.count += 1
                        if count >= self.threshold:
                            sta.count_filter += 1
                is_most_common = False
        result_text.append("")
        with open(f"output/{self.output_filename}.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(result_text))
        return result

    def print_info(self):
        type_enum = [["Failure-revealing", self.tp], ["False Positive", self.fp], ["Pass Correct", self.tn],
                     ["Pass Wrong", self.fn]]
        text_lines = ["---Single Depth Statistics---"]
        for type_name, type_class in type_enum:
            text_lines.append(f"{type_name}: {type_class.count}")
            if type_class.count > 0:
                text_lines.append(f"After count filter: {type_class.count_filter} ({type_class.count_filter / type_class.count*100:.2f}%)")
        text = "\n".join(text_lines)
        print(text)
        with open(f"output/statistics/{self.output_filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)


class FunctionExecute:
    def __init__(self, func_name):
        self.function_name = func_name
        self.output_collections = {}

    @func_set_timeout(120)
    def execute(self, module, argument: str):
        """
        :param module: code path without suffix
        :param argument: one piece of argument
        """
        with open(f"{module}.txt", "r", encoding="utf-8") as f:
            code = f.read()
        globals_dict = {}
        exec(code, globals_dict)
        my_function = globals_dict[self.function_name]
        eval_args = eval(argument)
        if isinstance(eval_args, tuple) and (argument[0] != "(" or argument[-1] != ")"):
            return my_function(*eval_args)
        else:
            return my_function(eval_args)

    def iterate_module(self, target_file_path_list, subject, arguments_list_arg: str=None):
        '''
        @param
        generated_input: arguments are generated from LLM, and hence argument_list should be passed from mutated input
        '''
        file_name_list = []
        for filename in target_file_path_list:
            file_name_list.append(filename.split("/")[-1].replace("txt", "py"))
        
        arguments_list = None
        # print(f"arguments_list in iterate_module:{arguments_list}")
        if arguments_list_arg is None:
            with open(f"{root}/arguments/{subject}.txt", encoding="utf-8") as f:
                arguments_list = eval(f.read())
        else:
            print("Use self-defined arguments.")
            arguments_list = eval(arguments_list_arg)
        # print(f"arguments_list_arg after definition:{arguments_list_arg}")
        # print(f"arguments_list in iterate_module after definition:{arguments_list}")
        # assert False, "debugging"
        board = StatisticsBoard()
        cut_result_list = []
        correct_result_list = []
        # print(f"!!!length of arguments_list:{len(eval(arguments_list))}")
        # print(f"!!!arguments_list[0][0]:{eval(arguments_list)[0][0]}")
        pbar = tqdm(total=len(arguments_list))
        for index, argument in enumerate(arguments_list):
            suffix = "_" + subject.split("_")[-1]
            try:
                correct_result_list.append(
                    [index, self.execute(f"{root}/correct_code/{subject.removesuffix(suffix)}", argument)])
            except Exception:
                if arguments_list_arg is not None:
                    continue
                else:
                    print("Correct code raise Exception, check bugs manually.")
                    sys.exit(1)
            for module in target_file_path_list:
                if module in list(self.output_collections):  # if the identifier has been executed before
                    if index < len(self.output_collections[module]):
                        board.add_result(index, self.output_collections[module][index], module)
                        continue
                try:
                    output_filename_without_suffix = module.removesuffix(".txt")
                    program_result = self.execute(output_filename_without_suffix, argument)
                except FunctionTimedOut:
                    print(module, "timeout!")
                    program_result = "Exception"
                except Exception:
                    program_result = "Exception"
                board.add_result(index, program_result, module)
                if module not in list(self.output_collections):
                    self.output_collections[module] = []
                self.output_collections[module].append(program_result)
            try:
                cut_result = self.execute(get_target_code_path(subject).removesuffix(".txt"), argument)
            except FunctionTimedOut:
                print(f"CUT of {subject} timeout!")
                cut_result = "Exception"
            except Exception:
                cut_result = "Exception"
            cut_result_list.append([index, cut_result])
            pbar.update(1)
        pbar.close()
        return board, cut_result_list, correct_result_list


class ClassExecute:
    def iterate_module(self, target_file_path_list, subject, arguments_list_arg:list[str]=None):
        pbar = tqdm(total=len(target_file_path_list) + 2)
        board = StatisticsBoard()
        for file in target_file_path_list:
            file_without_suffix = file.removesuffix(".txt")
            rv_result = class_execute(file_without_suffix, subject, arguments_list_arg)
            for result_line in rv_result:
                try:
                    board.add_result(result_line[0], result_line[1], file)
                except Exception:
                    board.add_result(result_line[0], "Exception", file)
            pbar.update(1)
        cut_result_list = class_execute(get_target_code_path(subject).removesuffix(".txt"), subject, arguments_list_arg)
        pbar.update(1)
        suffix = "_" + subject.split("_")[-1]
        correct_result_list = class_execute(f"{root}/correct_code/{subject.removesuffix(suffix)}", subject, arguments_list_arg)
        is_return_result = False
        for correct_result in correct_result_list:
            if correct_result[1] and correct_result[1] != "Exception":
                is_return_result = True
                break
        if not is_return_result:
            raise ValueError("Don't forget to output the result!")
        pbar.update(1)
        pbar.close()
        return board, cut_result_list, correct_result_list


def calculate_max_depth(identifiers):
    max_depth = 0
    for identifier in identifiers:
        if identifier.depth.isnumeric() and int(identifier.depth) >= max_depth:
            max_depth = int(identifier.depth)
    return max_depth


def check_file(output_filename):
    if os.path.isfile(f"output/{output_filename}.txt"):
        print(f"File {output_filename} already exists. Are you sure you want to overwrite it? y/n")
        choice = input()
        if choice == 'y':
            with open(f"output/{output_filename}.txt", "w", encoding="utf-8") as f:
                f.write("")
        elif choice == 'n':
            print("Please enter a new output filename:")
            output_filename = input()
    return output_filename


def convert_list_to_str(string: str):
    list_str = string.removeprefix("[").removesuffix("]")
    ele_str = list_str.split(",")
    result = []
    for ele in ele_str:
        result.append(ele.strip())
    return result


def count_most_common(output_dict):
    result_list = []
    for file, output in output_dict.items():
        is_exist = False
        index = 0
        for i, result in enumerate(result_list):
            if output == result[0]:
                is_exist = True
                index = i
        if not is_exist:
            result_list.append([output, list(output_dict.values()).count(output), [file]])
        else:
            result_list[index][2].append(file)
    result_list.sort(key=lambda x: x[1], reverse=True)
    return result_list


def create_node_lines(start, end):
    result = []
    for i in range(start, end + 1):
        result.append(i)
    return result


def class_execute(module, subject: str, test_cases=None) -> list[list]:
    """
    :param module: path of code to be tested without suffix (e.g. "mutation/subjects/target_code/classeval_xxx_1")
    :param subject: subject name (e.g. "classeval_xxx_1")
    :param test_cases: "[[Class(...), function1(...), ...], [...], ...]"
    :return: [[output_idx, function_output], ...]
    """
    with open(f"{module}.txt", "r", encoding="utf-8") as f:
        code = f.read()
    suffix = "_" + subject.split("_")[-1]
    with open(f"{root}/test_input/{subject.removesuffix(suffix)}.txt", 'r', encoding="utf-8") as f:
        base_code = f.read()
    if test_cases:
        output = []
        for test_case in convert_list_to_str(test_cases):
            output.append(utils.class_execute(code, convert_list_to_str(test_case), 0))
        return output
    else:
        if "<generated_code_here>" in base_code:
            infilled_code = base_code.replace("<generated_code_here>", code)
        else:
            print("<generated_code_here> not found. Infilling would be to no avail!")
            raise KeyError
        globals_dict = {}
        try:
            exec(infilled_code, globals_dict)
        except Exception:
            raise Exception(f"Reports a exec error from {module}")
        my_function = globals_dict['obtain_output']
        try:
            return my_function()
        except FunctionTimedOut:
            return [[0, "Exception"]]
        except Exception as e:
            print(e)
            return [[0, "Exception"]]


def find_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value]


def main(subject_arg=None,subject_type_arg=None,model_arg=None,verification_type_arg=None,function_name_arg=None,argument_list_arg=None):
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/statistics", exist_ok=True)
    '''
    @param
    generated_input:using input generated by LLM, in this case the filename of output should change a little bit
    '''
    #python CrossVerificationDemo.py evo14 function qwen cv all_prefix_suffix_pairs
    #python CrossVerificationDemo.py evo27 function starchat sd flip_case_special
    #python CrossVerificationDemo.py evo11c function starchat sd string_xor_advanced
    if subject_arg == None:
        subject = sys.argv[1]
    else:
        subject = subject_arg
    
    if subject_type_arg == None:
        subject_type = sys.argv[2]
    else:
        subject_type = subject_type_arg

    if model_arg == None:
        model = sys.argv[3]
    else:
        model = model_arg

    if verification_type_arg == None:
        verification_type = sys.argv[4]
    else:
        verification_type = verification_type_arg

    if verification_type == "cv":
        sta_class = CrossVerificationDemo
    elif verification_type == "cvd":
        sta_class = CrossVerificationDiscount
    elif verification_type == "dp":
        sta_class = DifferentialPrompting
    elif verification_type == "sd":
        sta_class = SingleDepth
    else:
        print("Invalid verification type")
        sys.exit(1)
    if subject_type == "function":
        # argument_list_arg has not been provided to the main function
        if function_name_arg == None:
            args = load_args(subject)
            function_name_arg = args["func_name"]
            sta = sta_class(FunctionExecute(function_name_arg), subject, model)
        else:
            # argument_list_arg has been provided to the main function
            print(f"argument_list_arg provided.")
            print(f"argument_list_arg:{argument_list_arg}")
            sta = sta_class(FunctionExecute(function_name_arg), subject, model,argument_list=argument_list_arg)
    elif subject_type == "class":
        if argument_list_arg:
            sta = sta_class(ClassExecute(), subject, model, argument_list_arg)
        else:
            sta = sta_class(ClassExecute(), subject, model)
    else:
        print("Invalid subject type")
        sys.exit(1)

    #A technique to reduce FP
    docstring_verification = False

    sta.iterate(docstring_verification)
    sta.print_info()

if __name__ == "__main__":
    main()

