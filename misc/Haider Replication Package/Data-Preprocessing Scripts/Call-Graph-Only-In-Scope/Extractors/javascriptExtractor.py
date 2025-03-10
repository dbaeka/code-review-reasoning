from Extractors.globalVariables import *
def extractTreeJS(node, info):                                        #javascript extraction
    # print(node)
    # print(node.children)
    if node.type == 'method_definition' or node.type == 'program' or node.type == 'class_declaration':
        info.functions.append(node)
        if node.type == 'method_definition' or node.type == 'class_declaration':
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type.find('identifier') >= 0:
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
            info.all.add(temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreeJS(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'member_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        if node.children[0].type == 'identifier':
            addCall(info.function_names[-1], node.children[2].text.decode('utf-8').split('(')[0], info)
            for child in node.children[1:]:
                extractTreeJS(child, info)
            return
    

    if node.type == 'call_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        if node.children[0].type == 'identifier':
            addCall(info.function_names[-1], furnish_variable_name(node.children[0].text.decode('utf-8')), info)
            for child in node.children[1:]:
                extractTreeJS(child, info)
            return
    

    for child in node.children:
        extractTreeJS(child, info)    
