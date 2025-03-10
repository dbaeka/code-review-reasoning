import json
import sys

filename = sys.argv[1]
file = open(filename, 'r')
data = []
for line in file:
    data.append(json.loads(line))
file.close()

file = open(filename, 'w')
for i in range(len(data)):
    ret = data[i]['result']
    ans = ''
    for x in ret.split(' '):
        if x != '':
            ans += x + ' '
    if ans != '':
        ans = ans[:-1]
    data[i]['result'] = ans
    file.write(json.dumps(data[i]) + '\n')
file.close()