import random

default = ['Taoli', 'Zijing', 'Dingxiang', 'Tingtao', 'Guanchou', 'Qingfen', 'Heyuan']

def initialize(list = default, i = 3, file1 = 'record.txt', n = 6, day = 1):
    with open(file1, 'w') as f:
        f.write('result: '+str(i)+' count: '+str(day)+'\n')
        for k in range(n+1):
            f.write(list[k])
            f.write(' ')
        f.write('\n')

def record(d, file1 = 'record.txt'):
    n = d[0]['count']
    with open(file1, 'w') as f:
        for i in range(n):
            f.write('result: '+str(d[i]['result'])+' count: '+str(d[i]['count'])+'\n')
            for k in range(len(d[i]['list'])):
                f.write(d[i]['list'][k]+' ')
            f.write('\n')

def decision(list = default, file1 = 'record.txt', file2 = 'record.txt'):
    f1 = open(file1, 'r')
    a = f1.readline().split()
    re1 = int(a[1])
    count = int(a[3])
    reference = f1.readline().split()
    n = len(list)
    index = []
    for i in range(n):
        if list[i]!=reference[re1]:
            index.append(i)
    re2 = index[random.randrange(len(index))]
    print(list[re2])
    print('This is the '+str(count+1)+'th resolution.')
    out = []
    out.append({})
    out.append({})
    out[0]['list'] = list
    out[0]['count'] = count+1
    out[0]['result'] = re2
    out[1]['list'] = reference
    out[1]['count'] = count
    out[1]['result'] = re1
    for j in range(count-1):
        out.append({})
        a = f1.readline().split()
        out[j+2]['result'] = int(a[1])
        out[j+2]['count'] = int(a[3])
        out[j+2]['list'] = f1.readline().split()
    f1.close()
    record(out, file2)

if __name__=="__main__":
	decision()


