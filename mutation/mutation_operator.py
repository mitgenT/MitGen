def define_specific_mutation(indent:str) -> list[str]:
    return [\
    f"{indent}if <MASK>:\n{indent}    pass\n{indent}else:\n{indent}    <MASK>\n",\
    f"{indent}if <MASK>:\n{indent}    <MASK>\n",\
    f"{indent}while <MASK>:\n{indent}    <MASK>\n",\
    f"{indent}return <MASK>\n"]

def add_specific(line:str,indent:str,option:str,keywords_cannot_perform_specific_mutation:list[str]) -> (str,str):
    # print(f"Performing mutation operator: {option}...")
    # 'code = f"{indent}if <missing code>:\\n{indent}{indent}pass\\n{indent}else:\\n{indent}{indent}<missing code>\\n"',\
    # 'code = f"{indent}if <missing code>:\\n{indent}{indent}<missing code>\\n"',\
    # 'code = f"{indent}while <missing code>:\\n{indent}{indent}<missing code>\\n"',\
    # 'code = f"{indent}return <missing code>\\n"'
    # if not any([ keywords in line for keywords in keywords_cannot_perform_specific_mutation]):
    code = option
        # print(f"code: {code}")
    # return mutated code and mutation label for naming files
    return code,"add_specific"


def simply_replace_missing_line(line:str,indent:str) -> (str,str):
    # print(f"Performing mutation operator: Simply replacing...")
    code_to_replace = f"{indent}<MASK>"
    #return mutated code and mutation label for naming files
    return code_to_replace, "simply_replace"

def simply_insert_missing_line(line:str,indent:str) -> (str,str):
    # print(f"Performing mutation operator: Simply adding...")
    code_to_insert = f"{indent}<MASK>\n{indent}{line}"
    return code_to_insert, "simply_insert"


