import ast
import sys

from mutation.utils import get_target_code_path


# def print_ast_tree(node, indent=0):
#     print('  ' * indent, node.__class__.__name__)
#     for child in ast.iter_child_nodes(node):
#         print_ast_tree(child, indent + 1)

# def flatten(S):
#     if S == []:
#         return S
#     if isinstance(S[0], list):
#         return flatten(S[0]) + flatten(S[1:])
#     return S[:1] + flatten(S[1:])
class AstMutator:
    def __init__(self, input_mode, func_name=None):
        self.input_mode = input_mode
        if self.input_mode == "function":
            self.func_name = func_name
            if self.func_name is None:
                raise Exception("No func_name")

    def get_nodes_data(self, node, curr_depth):
        output_lines = []
        for child in ast.iter_child_nodes(node):
            # print(f"child:{child}")
            first_con = isinstance(child,ast.stmt) and not isinstance(child, ast.FunctionDef) and not (isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant)) and not isinstance(child, ast.Import) and not isinstance(child, ast.ImportFrom) and not isinstance(child, ast.ClassDef)
            second_con = self.input_mode != "function" or not isinstance(child, ast.FunctionDef) or isinstance(child, ast.FunctionDef) and child.name == self.func_name
            # print(f"first_con:{first_con}")
            # print(f"second_con:{second_con}")
            if (isinstance(child,ast.stmt) and not isinstance(child, ast.FunctionDef) and not (isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)) and not isinstance(child, ast.Import)
                    and not isinstance(child, ast.ImportFrom) and not isinstance(child, ast.ClassDef)
                    and not (isinstance(node, ast.FunctionDef) and child in node.decorator_list)):
                data = f"depth:{curr_depth},node:{child},start_line:{child.lineno},end_line:{child.end_lineno}"
                # print('  ' * (curr_depth-1)*2,data)
                output_lines.append(data)
            if (self.input_mode != "function" or not isinstance(child, ast.FunctionDef) or
                    isinstance(child, ast.FunctionDef) and child.name == self.func_name):
                returned_nodes = self.get_nodes_data(child, curr_depth + 1)
            else:
                returned_nodes = []
            if returned_nodes != []:
                output_lines= output_lines + returned_nodes
        return output_lines

    def load_and_run(self, input_program):
        """
        return a dictionary, key: depth, value: line of blocks (e.g. [[2, 7], [8, 12], [14, 14]])
        """
        # input_program = ""
        # with open(f'{utils.root}/target_code/{input_program}', encoding="utf-8") as f:
        #     input_program = f.read()
        # start_line = 0
        # end_line = len(input_program)
        parsed_ast = ast.parse(input_program)
        # print(ast.dump(parsed_ast, include_attributes=True, indent=3))
        # print_ast_tree(parsed_ast)
        if self.input_mode == "class":
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.ClassDef):
                    output_lines = self.get_nodes_data(node, 1)
                    break
        else:
            output_lines = self.get_nodes_data(parsed_ast, 1)
        max_depth = get_max_depth(output_lines)
        depth_to_line_dict = {}
        for depth in range(1, max_depth + 1):
            # line_to_mask is for debugging, merged_line_to_mask should be the final output
            line_to_mask, merged_line_to_mask = get_lines_for_masking(output_lines, depth)
            depth_to_line_dict[depth] = merged_line_to_mask
        # print(f"line_to_mask: {line_to_mask}")
        # print(f"merged_line_to_mask: {merged_line_to_mask}")
        return depth_to_line_dict

def merge_single_stmts_to_block(line_to_mask):
    '''
    merge block aims to merge multiple single lines ast node into one block
    Detection of single line ast node is done by `each_line[0] == each_line[1]`
    consecutive single lines keep being inserted into merge block
    When `each_line[0] != each_line[1]`, a block will be created from the merge block, and the merge block will be re-initiated as an empty list
    '''
    merge_block = []
    output_block = []
    for idx, each_line in enumerate(line_to_mask):
        #each_line[0] is start line of a block, each_line[1] is end line of a block, start_line == end_line, which means the block is a single line
        if each_line[0] == each_line[1] : 
            #handle the case that there is a line break between two consecutive single line ast node, in this case just flush the current merge block and then create a new one
            #merely compare indentation is not enough, because we only consider ast.stmt, which will skip function signature
            if merge_block != [] and merge_block[-1] != each_line[0] - 1:
                block_start_line = min(merge_block)
                block_end_line = max(merge_block)
                output_block.append([block_start_line,block_end_line])
                merge_block = []
            merge_block.append(each_line[0])
            
            # The current line is the last line
            if idx == len(line_to_mask) - 1:
                block_start_line = min(merge_block)
                block_end_line = max(merge_block)
                output_block.append([block_start_line,block_end_line])
            # print(f"output_block in 1:{output_block}")
            continue
        elif merge_block != []:
            block_start_line = min(merge_block)
            block_end_line = max(merge_block)
            output_block.append([block_start_line,block_end_line])
            merge_block = []
        output_block.append(each_line)
        # print(f"output_block in 3:{output_block}")
        
    return output_block


def get_lines_for_masking(output_lines,target_depth):
    line_to_mask = []
    for each_line_data in output_lines:
        line_depth = each_line_data.split(",")[0].split(":")[1]
        start_line = ""
        end_line = ""
        # if 
        if int(line_depth) == target_depth:
            start_line = int(each_line_data.split(",")[2].split(":")[1])
            end_line = int(each_line_data.split(",")[3].split(":")[1])
            line_to_mask.append([start_line,end_line])
        # #Not the depth we are interested in
        # if int(line_depth) > target_depth:
        #     continue
        # else:
        #     #depth we are interested but not the last depth
        #     start_line = int(each_line_data.split(",")[2].split(":")[1])
        #     end_line = int(each_line_data.split(",")[2].split(":")[1])
        #     #last depth
        #     if int(line_depth) == target_depth:
        #         end_line = int(each_line_data.split(",")[3].split(":")[1])
        # line_to_mask.append([start_line,end_line])
    merged_line_to_mask = merge_single_stmts_to_block(line_to_mask)
    return line_to_mask, merged_line_to_mask

def get_max_depth(output_lines):
    max_depth = 0
    for each_line in output_lines:
        line_depth = each_line.split(",")[0].split(":")[1]
        if max_depth < int(line_depth):
            max_depth = int(line_depth)
    return max_depth


if __name__ == '__main__':
    subject = sys.argv[1]
    with open(get_target_code_path(subject), encoding="utf-8") as f:
        input_program = f.read()
    am = AstMutator("function")
    depth_to_line_dict = am.load_and_run(input_program)
    print(depth_to_line_dict)

# print("============Below shows the output lines============")
# print(output_lines)
# print(f"length of output lines: {len(output_lines)}")
# print(list(flatten(output_lines)))

#============================

# import ast
# import astunparse

# class ReplaceNode(ast.NodeTransformer):
#     def __init__(self, replace_index):
#         self.replace_index = replace_index
#         self.current_index = 0

#     def generic_visit(self, node):
#         # if self.current_index == self.replace_index:
#         if hasattr(node, 'body'):
#             print(f"node body: {node.body}")
#         # if isinstance(node,ast.For):
#             #  print(f"node lineno: {node.lineno}")
#             #  print(f"node end_lineno: {node.end_lineno}")
#             # print(f"node body: {node.body}")
#             # return ast.Constant(value="<missing code>")
#         # self.current_index += 1
#         return super().generic_visit(node)

# def replace_nodes_with_missing_code(input_code):
#     input_ast = ast.parse(input_code)
#     num_nodes = sum(1 for _ in ast.walk(input_ast))

#     output_programs = []
#     # for i in range(num_nodes):
#     transformer = ReplaceNode(replace_index=1)
#     #     print(f"target_ast: {ast.dump(input_ast)}")
#     modified_ast = transformer.visit(input_ast)
#     #     # print(f"modified_ast: {ast.dump(modified_ast)}")
#     modified_code = astunparse.unparse(modified_ast)
#     return modified_code
#     #     output_programs.append(modified_code)

#     return output_programs

# if __name__ == "__main__":
#     code_path = "subjects/target_code/916a.txt"
#     input_program = ""

#     with open(code_path, encoding="utf-8") as f:
#         input_program = f.read()

#     input_ast = ast.parse(input_program)
#     print(f"input_ast: {ast.dump(input_ast, include_attributes=True,indent=2)}")
#     output_programs = replace_nodes_with_missing_code(input_ast)
#     print("===================================")
#     print(f"output_programs: {output_programs}")
#     # print(f"{ast.dump(input_ast, include_attributes=True,indent=2)}")


#==================

    # for each_node in ast.walk(input_ast):
    #     print(f"each_node: {ast.dump(each_node, include_attributes=True)}")
    #     print("===========================")

    # output_programs = replace_nodes_with_missing_code(input_program)
    # for i, program in enumerate(output_programs):
    #     print(f"Program {i + 1}:\n{program}\n")
    #     break

# import ast

# class PrintNodes(ast.NodeVisitor):
#     def visit(self, node):
#         print(type(node).__name__)
#         self.generic_visit(node)

# code_path = "subjects/target_code/916a.txt"
# code_content = ""

# with open(code_path, encoding="utf-8") as f:
#     code_content = f.read()

# tree = ast.parse(code_content)

# print(ast.dump(tree, include_attributes=True, indent=3))

# PrintNodes().visit(tree)

# for each_node in ast.dump(tree):
#     print(each_node)