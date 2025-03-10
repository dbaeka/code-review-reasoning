from Extractors.globalVariables import *
def extractTreeCS(node, info):                                        #csharp extraction
    # print(node)
    # print(node.children)
    if node.type == 'method_declaration' or node.type == 'compilation_unit' or node.type == 'class_declaration' or node.type == 'struct_declaration':
        info.functions.append(node)
        if node.type == 'method_declaration' or node.type == 'class_declaration' or node.type == 'struct_declaration':
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type == 'identifier':
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            # print(temp)
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreeCS(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'invocation_expression' or node.type == 'member_access_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        t = 0
        for i in range(0,len(node.children)):
            if node.children[i].type.find('identifier') >= 0:
                t = i
            if node.children[i].type == 'argument_list':
                break
        if t < len(node.children) and node.children[t].type.find('identifier') >= 0:
            addCall(info.function_names[-1], furnish_variable_name(node.children[t].text.decode('utf-8')), info)
            for child in node.children[t + 1:]:
                extractTreeCS(child, info)
            return
    

    for child in node.children:
        extractTreeCS(child, info)    
