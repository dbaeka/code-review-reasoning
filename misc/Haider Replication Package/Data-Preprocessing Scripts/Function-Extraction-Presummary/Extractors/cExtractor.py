from Extractors.globalVariables import *
def extractTreeC(node, info):
    # print(node)
    if node.type == 'function_definition':
        if intersect(node.start_point, node.end_point, info.start, info.end):
            info.output += node.text.decode('utf-8')
            info.output += '\n'
        
        return

    for child in node.children:
        extractTreeC(child, info)    

