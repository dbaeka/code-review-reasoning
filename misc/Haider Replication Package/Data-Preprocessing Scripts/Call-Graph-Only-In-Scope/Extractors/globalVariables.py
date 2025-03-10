debug = 0
class Info:
    def __init__(self):
        self.functions = []
        self.function_names = []
        self.starts = []
        self.ends = []
        self.adjacency_list = {}
        self.scope_graph = {}
        self.stack = set()
        self.all = set()

def furnish_variable_name(name):
    name = name.split('(')[0]
    name = name.split('}')[-1]
    name = name.split('\n')[-1]  
    name = name.split(' ')[-1]
    name = name.split('[')[0]
    name = name.split('.')[-1]
    name = name.split('->')[-1]
    name = name.split('::')[-1]
    name = name.split('*')[-1]
    name = name.split('&')[-1]
    return name

def addCall(function_name, call_name, info):
    # print(function_name)
    # print(call_name)
    # if call_name not in lastScope:
    #     return
    if call_name not in info.all:
        return
    if function_name not in info.adjacency_list:
        info.adjacency_list[function_name] = set()
    name = call_name
    # if call_name in lastScope and len(lastScope[call_name]) > 0:
    #     name = lastScope[call_name][-1]
    info.adjacency_list[function_name].add(name)

