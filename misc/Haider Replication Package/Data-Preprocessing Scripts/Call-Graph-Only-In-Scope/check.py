count = 0
with open('result.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line == '[]\n':
            count += 1

print(count)

# import subprocess
# import os
# import ghlinguist as ghl

# def detect_language(code):
#     subprocess.run(['mkdir temp'], shell=True)
#     os.chdir('temp')
#     subprocess.run(['git init'], shell=True)
#     subprocess.run(['cat > temp.cpp'], shell=True, input=code.encode())
#     subprocess.run(['git add .'], shell=True)
#     subprocess.run(['git commit -m "temp"'], shell=True)
#     # result = subprocess.run(['github-linguist'], shell=True)  
#     result = ghl.linguist('.')
#     os.chdir('..')
#     subprocess.run(['rm -rf temp'], shell=True)
#     return result
#     # return None
#     # cmd = ['github-linguist', '-']
#     # result = subprocess.run(cmd, input=code.encode(), text=True, capture_output=True)
#     # return result.stdout.strip()

# code_snippet = open('sample.py', 'r').read()

# detected_language = detect_language(code_snippet)
# print(f"Detected language: {detected_language}")
# from tree_sitter import Language, Parser
# langs = ['python', 'go', 'java', 'c', 'cpp', 'c_sharp', 'ruby', 'php', 'javascript']
# code_snippet = open('test.java', 'r').read()
# for x in langs:
#     LANG = Language('build/my-languages.so', x)
#     parser = Parser()
#     parser.set_language(LANG)
#     tree = parser.parse(bytes(code_snippet, "utf-8"))
#     if tree.root_node.has_error:
#         continue
#     print(x)