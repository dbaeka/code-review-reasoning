from Extractors.globalVariables import *
def extractTreeRB(node, info):                        #ruby extraction      
    if node.type == 'method' or node.type == "singleton_method" :
        if intersect(node.start_point, node.end_point, info.start, info.end):
            info.output += node.text.decode('utf-8')
            info.output += '\n'
        
        return



    for child in node.children:
        extractTreeRB(child, info)    
