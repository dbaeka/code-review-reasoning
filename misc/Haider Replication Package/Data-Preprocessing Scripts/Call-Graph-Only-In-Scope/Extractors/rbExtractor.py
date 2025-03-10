from Extractors.globalVariables import *
def extractTreeRB(node, info):                        #ruby extraction  
    # print(node)                                      
    # for i in range(0,len(node.children)):
    #     if node.children[i].type == '.':
    #         print(node.children[i+1].text.decode('utf-8') + " " + str(i) + " " + node.type)            
    if node.type == 'method' or node.type == 'program' or node.type == "class" or node.type == "singleton_method" :
        info.functions.append(node)
        if node.type == 'method' or node.type == "class" or node.type == "singleton_method":
            temp = ''
            for i in range(0,len(node.children)):
                if node.children[i].type == 'identifier' or node.children[i].type == 'constant':
                    temp = furnish_variable_name(node.children[i].text.decode('utf-8'))
                    break
            if temp == '':
                for child in node.children:
                    extractTreeRB(child, info)    
                return
            info.function_names.append(info.function_names[-1] + "#::#" + temp)
            info.all.add(temp)
        else:
            info.function_names.append('')
        info.starts.append(node.start_point)
        info.ends.append(node.end_point)
        for child in node.children:
            extractTreeRB(child, info)
        info.functions.pop()
        info.function_names.pop()
        info.starts.pop()
        info.ends.pop()
        return


    if node.type == 'call':
        assert info.starts[-1] <= node.start_point and info.ends[-1] >= node.end_point
        t = 0
        for i in range(0,len(node.children)):
            if node.children[i].type.find('identifier') >= 0:
                t = i
            if node.children[i].type == 'argument_list':
                break
        if node.children[t].type.find('identifier') >= 0:
            addCall(info.function_names[-1], furnish_variable_name(node.children[t].text.decode('utf-8')), info)
            for child in node.children[1:]:
                extractTreeRB(child, info)
            return
    

    for child in node.children:
        extractTreeRB(child, info)    
