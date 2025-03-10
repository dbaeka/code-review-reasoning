from Extractors.globalVariables import *

def extractTreeJV(node, info):                                        #java extraction
    # print(node)
    # print(node.children)
    if node.type == 'method_declaration' or node.type == 'program' or node.type == 'class_declaration':
        info.functions.append(node)
        if node.type == 'method_declaration' or node.type == 'class_declaration':
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
            extractTreeJV(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return

    if node.type == 'method_invocation' or node.type == 'object_creation_expression':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        # print(node.children)
        t = 0
        for i in range(0,len(node.children)):
            if node.children[i].type.find('identifier') >= 0:
                t = i
            if node.children[i].type == 'argument_list':
                break
        if node.children[t].type.find('identifier') >= 0:
            addCall(info.function_names[-1], furnish_variable_name(node.children[t].text.decode('utf-8')), info)
            for child in node.children[1:]:
                extractTreeJV(child, info)
            return
    

    for child in node.children:
        extractTreeJV(child, info)    
