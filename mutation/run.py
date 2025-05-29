from datetime import timedelta
from utils import load_args, root
import perform_mutation as pm
import generate_code_with_llm as gcllm
import sys
import time
'''
This program performs mutation + code generation. Please run the individual programs for individual functions.
'''

if __name__ == '__main__':
    target = sys.argv[1] #e.g., 886d.txt
    model = sys.argv[2] #e.g., gpt or gemini
    mode = sys.argv[3] #e.g., single or ast
    input_mode = sys.argv[4]
    func_name = None
    dp_mode = False
    if input_mode == "function":
        args = load_args(target.removesuffix(".txt"))
        func_name = args["func_name"]
    if len(sys.argv) > 5:
            dp_mode = sys.argv[5]
    pm.main(target,model, func_name, mode, input_mode)
    start_time = time.time()
    gcllm.main(target,model,mode, dp_mode, func_name)
    if not dp_mode:
        elapsed_time = time.time() - start_time
        with open(f"{root}/time/{model}_{target}", "a", encoding="utf-8") as time_file:
            time_file.write(f'elapsed time: {timedelta(seconds=elapsed_time)}\n')