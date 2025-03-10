from Extractors.globalVariables import *
def extractTreeC(node, info):                                   #c extraction
    # print(node)

    if node.type == 'function_definition' or node.type == 'translation_unit':
        info.functions.append(node)    
        if node.type == 'function_definition':
            temp = furnish_variable_name(node.children[1].text.decode('utf-8'))
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreeC(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'call_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        if node.children[0].type == 'identifier':
            addCall(info.function_names[-1], furnish_variable_name(node.children[0].text.decode('utf-8')), info)
            for child in node.children[1:]:
                extractTreeC(child, info)
            return
    

    for child in node.children:
        extractTreeC(child, info)    

