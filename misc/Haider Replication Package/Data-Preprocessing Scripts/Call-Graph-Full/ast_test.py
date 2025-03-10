from tree_sitter import Language, Parser
import subprocess
import sys
import json
import argparse

from Extractors.globalVariables import *
from Extractors.pythonExtractor import extractTreePY 

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

filename = 'test.py'

code = open(filename, 'r').read()
parser = Parser()
parser.set_language(PY_LANGUAGE)
tree = parser.parse(bytes(code, 'utf8'))

#display tree with name of node
def display_tree(node, indent=0):
    print(' ' * indent + node.type, end='')
    if len(node.children) == 0:
        print(' ' + node.text.decode('utf-8'), end='')
    print()
    for child in node.children:
        display_tree(child, indent + 2)



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

info = Info()
display_tree(tree.root_node)
extractTreePY(tree.root_node, info)
buildScopeGraph(info)
print(dfs('','' , info)[2:])