from Extractors.globalVariables import *
def extractTreePHP(node, info):                            
    if node.type == 'method_declaration' or node.type == 'program' or node.type == 'class_declaration' or node.type == 'function_definition':
        info.functions.append(node)
        if node.type == 'method_declaration' or node.type == 'class_declaration' or node.type == 'function_definition':
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type == 'name':
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            # print(temp)
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreePHP(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'member_call_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        t = 0
        while(t < len(node.children) and node.children[t].type != 'name'):
            t += 1
        if t < len(node.children) and node.children[t].type == 'name':
            addCall(info.function_names[-1], furnish_variable_name(node.children[t].text.decode('utf-8')), info)
            for child in node.children[t + 1:]:
                extractTreePHP(child, info)
            return
    

    for child in node.children:
        extractTreePHP(child, info)    
