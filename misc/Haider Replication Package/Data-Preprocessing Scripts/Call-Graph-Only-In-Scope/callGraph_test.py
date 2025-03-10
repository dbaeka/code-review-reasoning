from tree_sitter import Language, Parser
import subprocess
import sys
import json
import argparse
from Extractors.globalVariables import *
from Extractors.cppExtractor import extractTreeCPP 
from Extractors.csharpExtractor import extractTreeCS 
from Extractors.cExtractor import extractTreeC 
from Extractors.pythonExtractor import extractTreePY 
from Extractors.javaExtractor import extractTreeJV 
from Extractors.javascriptExtractor import extractTreeJS 
from Extractors.goExtractor import extractTreeGO 
from Extractors.rbExtractor import extractTreeRB
from Extractors.phpExtractor import extractTreePHP


langs = ['java', 'python', 'go',  'cpp', 'c', 'c_sharp', 'ruby', 'php', 'javascript']
# sys.setrecursionlimit(100000) 
GO_LANGUAGE = Language('build/my-languages.so', 'go')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
PY_LANGUAGE = Language('build/my-languages.so', 'python')
JV_LANGUAGE = Language('build/my-languages.so', 'java')
C_LANGUAGE = Language('build/my-languages.so', 'c')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
RB_LANGUAGE = Language('build/my-languages.so', 'ruby')
CS_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')

parser = argparse.ArgumentParser()
    
## Required parameters  
parser.add_argument("--file_path", default=None, type=str, required=True,
                        help=".jsonl/.c/.cpp...")
parser.add_argument("--output", default=None, type=str, required=False,
                        help=".output") 
parser.add_argument("--testcase", default=1000000, type=int, required=False,
                        help="number of case")
parser.add_argument("--debug", default=0, type=int, required=False,
                        help="debug mode")
parser.add_argument("--lang", default=None, type=str, required=False,
                        help="language")
args = parser.parse_args()    

debug = args.debug


def detect_language(code):
    for x in langs:
        LANG = Language('build/my-languages.so', x)
        parser = Parser()
        parser.set_language(LANG)
        try:
            tree = parser.parse(bytes(code, "utf-8"))
        except:
            continue
        if tree.root_node.has_error:
            continue
        return x
    return None



def generate_call_graph(source_code, language, info):
    parser = Parser()
    
    if language == 'python':
        parser.set_language(PY_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreePY(root_node, info)

    elif language == 'javascript':
        parser.set_language(JS_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeJS(root_node, info)
        
    elif language == 'go':
        parser.set_language(GO_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeGO(root_node, info)
        
    elif language == 'java':
        parser.set_language(JV_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeJV(root_node, info)

    elif language == 'c':
        parser.set_language(C_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeC(root_node, info)

    elif language == 'cpp':
        parser.set_language(CPP_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeCPP(root_node, info)

    elif language == 'ruby':
        parser.set_language(RB_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeRB(root_node, info)

    elif language == 'c_sharp':
        parser.set_language(CS_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreeCS(root_node, info)
    
    elif language == 'php':
        parser.set_language(PHP_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return
            else:
                generate_call_graph(source_code, language, info)
                return
        extractTreePHP(root_node, info)

    else:
        print("Language not supported")
        exit(1)

def langConverter(lang):
    if lang.startswith('py') or lang.startswith('.py'):
        return 'python'
    elif lang.startswith('js') or lang.startswith('javascript') or lang.startswith('.js'):
        return 'javascript'
    elif lang.startswith('go') or lang.startswith('.go'):
        return 'go'
    elif lang.startswith('java') or lang.startswith('.java'):
        return 'java'
    elif lang.startswith('cp') or lang.startswith('c+') or lang.startswith('.cpp'):
        return 'cpp'
    elif lang.startswith('cs') or lang.startswith('c#') or lang.startswith('c_sharp') or lang.startswith('c-sharp') or lang.startswith('.cs'):
        return 'c_sharp'
    elif lang.startswith('c') or lang.startswith('.c'):
        return 'c'
    elif lang.startswith('rb') or lang.startswith('ruby') or lang.startswith('.rb'):
        return 'ruby'
    elif lang.startswith('php') or lang.startswith('.php'):
        return 'php'
    else:
        return None


def buildScopeGraph(info):
    for x in info.adjacency_list:
        lists = x.split('#::#')
        for i in range(0, len(lists) - 1):
            if lists[i] not in info.scope_graph:
                info.scope_graph[lists[i]] = set()
            info.scope_graph[lists[i]].add(lists[i+1])

def dfs(node, path, info):
    info.stack.add(node)
    ret = node + '->['
    if node in info.scope_graph:
        for x in info.scope_graph[node]:
            if x in info.stack:
                continue
            ret += dfs(x, path + '#::#' + x, info)
            ret += ','
    if path in info.adjacency_list:
        for x in info.adjacency_list[path]:
            ret += x
            ret += ','
    if ret[-1] == ',':
        ret = ret[:-1]
    info.stack.remove(node)
    return ret + ']'
    


def getCallGraph(code, lang):
    info = Info()
    generate_call_graph(code, lang, info)
    buildScopeGraph(info)
    return dfs('', '', info)[2:]

def removeEscapeCharacter(code):
    return bytes(code, "utf-8").decode("unicode_escape")
if __name__ == '__main__':
    if debug:
        filename = args.file_path

        code = open(filename, 'r').read()
        lang = args.lang
        if lang is None:
            lang = detect_language(code)
        if lang is None:
            print('Language not supported')
            exit(1)
        print(getCallGraph(code, lang).replace('->', ' -> ').replace(',', ', '))
        exit(0)



    file_path = args.file_path
    file = open(file_path, 'r')
    def read_one_test():
        code = file.readline()
        return code
    res = sys.stdout
    if args.output:
        res = open(args.output, 'w')
    testcaseCount = 0
    while testcaseCount < args.testcase:
        jsonObject = read_one_test()
        # print(jsonObject)
        if jsonObject:
            testcaseCount += 1
            # if  testcaseCount <= 46014: 
            #     continue
            jsonOb = json.loads(jsonObject)
            code = jsonOb['oldf']
            # print(code)
            lang = ''
            try:
                lang = jsonOb['lang']
            except:
                # for x in jsonOb:
                #     print(x)
                # lang = guess_lexer(code).name
                lang = detect_language(code)
                # print(lang)
                if lang is None:
                    lang = 'python'
                
            print('Test: ', testcaseCount, ' (', lang, ')            ', end='\r')
            # res.write(code)
            lang = langConverter(lang)
            # code = removeEscapeCharacter(code)
            callgraph = ''
            # try:

            callgraph = getCallGraph(code, lang)
            callgraph = callgraph.replace('\n', ' ')
            # print("done")
            # except:
            #     callgraph = '[]'
            #     print(lang)
            if callgraph == '[]':
                callgraph = '[No CFG could be retrieved]'
            res.write(json.dumps({'callgraph': callgraph}) + '\n')
            # print(callgraph)
            # res.write("\n\n\n")
            # print(summary)
        else:
            break

    res.close()
    print('Done')