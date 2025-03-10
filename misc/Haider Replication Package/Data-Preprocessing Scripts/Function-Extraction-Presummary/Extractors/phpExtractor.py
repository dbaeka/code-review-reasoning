from Extractors.globalVariables import *
def extractTreePHP(node, info):                            
    if node.type == 'method_declaration' or node.type == 'function_definition':
        if intersect(node.start_point, node.end_point, info.start, info.end):
            info.output += node.text.decode('utf-8')
            info.output += '\n'
        
        return
    

    for child in node.children:
        extractTreePHP(child, info)    
