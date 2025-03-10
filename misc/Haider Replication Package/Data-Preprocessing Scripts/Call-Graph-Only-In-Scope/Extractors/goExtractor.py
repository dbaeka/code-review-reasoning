from Extractors.globalVariables import *
def extractTreeGO(node, info):                                        #go extraction
    # print(node)
    if node.type == 'method_declaration' or node.type == 'source_file':
        info.functions.append(node)
        if node.type == 'method_declaration':
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type == 'field_identifier':
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            # print(temp)
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
            info.all.add(temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreeGO(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'call_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        t = 0
        # print(node.children)
        while(t < len(node.children) and node.children[t].type != 'selector_expression'):
            t += 1
        if t < len(node.children) and node.children[t].type == 'selector_expression':
            temp = node.children[t]
            t = 0
            while(t < len(temp.children) and temp.children[t].type != 'field_identifier'):
                t += 1
            if t < len(temp.children) and temp.children[t].type == 'field_identifier':
                addCall(info.function_names[-1], furnish_variable_name(temp.children[t].text.decode('utf-8')), info)
                
    

    for child in node.children:
        extractTreeGO(child, info)    
