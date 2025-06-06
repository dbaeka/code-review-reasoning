from Extractors.globalVariables import *
def extractTreePY(node, info):                                        #python extraction
    if node.type == 'function_definition' or node.type == 'module' or node.type == 'class_definition':
        info.functions.append(node)
        if node.type == 'function_definition' or node.type == 'class_definition':
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type == 'identifier':
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
            info.all.add(temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreePY(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'call':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        if node.children[0].type == 'identifier' or node.children[0].type == 'attribute':
            addCall(info.function_names[-1], furnish_variable_name(node.children[0].text.decode('utf-8')), info)
            for child in node.children[1:]:
                extractTreePY(child, info)
            return
    

    for child in node.children:
        extractTreePY(child, info)    
