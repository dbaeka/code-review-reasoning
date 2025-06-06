from Extractors.globalVariables import *
def extractTreeJS(node, info):                                        #javascript extraction
    # print(node)
    # print(node.children)
    if node.type == 'method_definition':
        if intersect(node.start_point, node.end_point, info.start, info.end):
            info.output += node.text.decode('utf-8')
            info.output += '\n'
        
        return
    

    for child in node.children:
        extractTreeJS(child, info)    
