from tree_sitter import Language, Parser
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


parser = argparse.ArgumentParser()
    
## Required parameters  
parser.add_argument("--file_path", default=None, type=str, required=True,
                        help=".jsonl/.c/.cpp...")
parser.add_argument("--output", default=None, type=str, required=True,
                        help=".output") 
parser.add_argument("--testcase", default=1000000, type=int, required=False,
                        help="number of case")
args = parser.parse_args()    


GO_LANGUAGE = Language('build/my-languages.so', 'go')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
PY_LANGUAGE = Language('build/my-languages.so', 'python')
JV_LANGUAGE = Language('build/my-languages.so', 'java')
C_LANGUAGE = Language('build/my-languages.so', 'c')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
RB_LANGUAGE = Language('build/my-languages.so', 'ruby')
CS_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')


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



def generate(source_code, language, info):
    parser = Parser()
    
    if language == 'python':
        parser.set_language(PY_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreePY(root_node, info)

    elif language == 'javascript':
        parser.set_language(JS_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeJS(root_node, info)
        
    elif language == 'go':
        parser.set_language(GO_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeGO(root_node, info)
        
    elif language == 'java':
        parser.set_language(JV_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeJV(root_node, info)

    elif language == 'c':
        # print("C")
        # print(source_code)
        parser.set_language(C_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        # print(root_node.type)
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeC(root_node, info)

    elif language == 'cpp':
        parser.set_language(CPP_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeCPP(root_node, info)

    elif language == 'ruby':
        parser.set_language(RB_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeRB(root_node, info)

    elif language == 'c_sharp':
        parser.set_language(CS_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreeCS(root_node, info)
    
    elif language == 'php':
        parser.set_language(PHP_LANGUAGE)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        if root_node.type == 'ERROR':
            language = detect_language(source_code)
            if language == None:
                return False
            else:
                return generate(source_code, language, info)
                
        extractTreePHP(root_node, info)

    else:
        print("Language not supported")
        exit(1)
    return len(info.output) > 0

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



def generateAround(source_code, info):
    tokenized = source_code.split('\n')
    assert info.start[0] >= 0 and info.start[0] < len(tokenized)
    cur = 0
    ret = ''
    info.end = (len(tokenized) - 1, len(tokenized[-1]))
    for i in range(info.start[0], len(tokenized)):
        if len(ret) == LEN:
            break
        if(len(ret) != 0):
            ret += '\n'
            cur += 1
        tokenized[i] = tokenized[i].replace('\t', ' ')
        tokenized[i] = tokenized[i].split(' ')
        temp = ''
        for j in range(len(tokenized[i])):
            tokenized[i][j] = tokenized[i][j].strip()
            if len(tokenized[i][j]) > 0:
                temp += tokenized[i][j] + ' '

        if cur + len(temp) > LEN:
            info.end = (i, LEN - cur)
            ret += temp[:LEN - cur]
            break
        cur += len(temp)
        ret += temp
        
    info.output = ret
    return ret, info.start[0], info.end[0]
    

def getResultAround(code, start, end):
    if start > 0:
        start -= 1
        end -= 1
    lo = max(0, start - LEN)
    hi = start
    tot = len(code.split('\n'))
    def util(code, mid):
        info = Info()
        info.start = (mid, 0)
        info.end = (mid, 0)
        generateAround(code, info)
        return info.output, info.start[0], info.end[0]


    while lo < hi:
        mid = (lo + hi) // 2
        ret, s, e = util(code, mid)
        if e < end:
            lo = mid + 1
        elif start - s > e - end and e < tot:
            lo = mid + 1
        else:
            hi = mid

    ret, s, e = util(code, lo)
    # print(s, e)
    return json.dumps({'result': ret}), s, e

def getResult(code, lang, start, end):
    presummary1, s1, e1 = getResultAround(code, start, end)

    info = Info()
    info.start = (start, 0)
    info.end = (end, 0)
    # print(start, end)
    if generate(code, lang, info) == False:
        return presummary1
    return json.dumps({'result': info.output})
    

def removeEscapeCharacter(code):
    return bytes(code, "utf-8").decode("unicode_escape")


def getRange(patch):
    # print(patch)
    
    tokens = patch.split(' ')
    tokens = tokens[1].split(',')
    if len(tokens) < 2:
        start = -int(tokens[0])
        end = start + 1
    else:
        start = -int(tokens[0])
        end = start + int(tokens[1])
    return start, end



if __name__ == '__main__':
    file_path = args.file_path
    file = open(file_path, 'r')
    def read_one_test():
        code = file.readline()
        return code
    
    output_filename = args.output
    res = sys.stdout
    if output_filename is not None:
        res = open(output_filename, 'w')
    
    
    testcaseCount = 0
    while testcaseCount < args.testcase:
        jsonObject = read_one_test()
        # print(jsonObject)
        if jsonObject:
            testcaseCount += 1
            jsonOb = json.loads(jsonObject)
            code = jsonOb['oldf']
            patch = jsonOb['patch']
            # print(code)
            lang = ''
            try:
                lang = jsonOb['lang']
            except:
                lang = detect_language(code)
                if lang is None:
                    lang = 'python'
                
            print('Test: ', testcaseCount, ' (', lang, ')            ', end='\r')
            # res.write(patch)
            # res.write(code)
            lang = langConverter(lang)
            
            start, end = getRange(patch)
            presummary = getResult(code, lang, start, end)
            # res.write("HERE\n")
            # res.write(json.loads(presummary)['result'])
            res.write(presummary)
            res.write("\n")
        else:
            break

    res.close()
    print('Done')
