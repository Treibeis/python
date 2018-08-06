import sys

def draw(PATTERN='*', MAX = 80):
    while 1:
        t1, t2 = 0, 0
        while t1==0:
            print('Please input the number of lines, n: ') # python3中print()是一个函数，它有一个参数是end，默认值为'\n'，所以是自动换行的，如果不想自动换行，可设置end=''。
            n=int(input())
            if n<1:
                print('\nInput error: n should be no less than 1.')
            else:
                t1=1
        while t2==0:
            print('Please input the number of spacings, m: ')
            m=int(input())
            if m<0:
                print('\nInput error: m should be no less than 0.')
            else:
                t2=1
        if 3*n+m-1>MAX:
            print('3n+m-1 should not be larger than',MAX)
            continue
        else:
            break
    for i in range(n):
        out = ' '*(n-i-1)+PATTERN*(2*i+1)+' '*m+PATTERN*(2*n-1-2*i)+' '*i
        print(out)
# 下面的代码是为命令行执行设计的。你把此python代码作为module导入的时候，并不会执行draw()，只有在命令行执行的时候，才会执行函数draw()。               
if __name__ == "__main__":
    draw()
    print("Press 'Enter' to exit.")
    a = input()
