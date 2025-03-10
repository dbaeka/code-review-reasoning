from Extractors.globalVariables import *
def extractTreeCS(node, info):                                        #csharp extraction
    if node.type == 'method_declaration':
        if intersect(node.start_point, node.end_point, info.start, info.end):
            info.output += node.text.decode('utf-8')
            info.output += '\n'
        
        return
    

    for child in node.children:
        extractTreeCS(child, info)    
