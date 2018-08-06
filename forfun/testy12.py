import time
import imp

dict1 ={1:31,2:29,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
dict2 ={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

def year(y):
    # Here is a correction (line 14-17).
    if y%400 == 0:
        return 1
    else:
        if y%4 == 0:
            if y%100 == 0:
                return 0
            else:
                return 1
        else:
            return 0

def day(y1 = 2017, m1 = 2, d1 = 24, y2 = 2017, m2 = 6, d2 = 4):
    t1 = time.time()
    # The code below is to rearrange the sequence of the two input dates if it's not correct.
    t = 0 
    if y1>y2:
        t = 1
    elif y1==y2:
        if m1>m2:
            t = 1
        elif m1==m2:
            if d1>d2:
                t = 1
    if t==1:
        temp = y1
        y1 = y2
        y2 = temp
        temp = m1
        m1 = m2
        m2 = temp
        temp = d1
        d1 = d2
        d2 = temp
    # Below is a refined implementation of your algorithm:
    # Pay attention to line 48-56 and 61-68, you have to update r1/r2 every time when m is changed, which is not considered in your old code. 
    # Note: Usually, a programm executes each line in order, any line that isn't in a circulation will be executed only once. In your old code, you assigin a value to r1/r2 before the circulation begins, so the value of r1/r2 would not vary according to the value of m in time.
    i = 0
    d = d1
    m = m1
    y = y1
    while y < y2:
        if year(y)==1:
                dc=dict1
            else:
                dc=dict2
        while m < 13:
            r1 = dc[m]
            # The for circulation below can be simply replaced by i += r1-d
            for a in range (d, r1):
                i += 1
            m += 1
            d = 0
        y += 1
        m = 1
    while m < m2:
        if year(y2)==1:
            r2=dict1[m]
        else:
            r2=dict2[m]
        for a in range (d, r2):
            i += 1
        m += 1
        d = 0
    while d < d2:
        i += 1
        d += 1
    t2 = time.time()
    return [i, t2-t1]
        

if __name__ =="__main__":
    # I placed the input command here so that you would not run them when you import you code into the python prompt.
    # To debug, you should open your python prompt and type 'import testy12 as T', then you can call the function day() simply by typing 'T.day(y1,m1,d1,y2,m2,d2)', here y1/2. m1/2, d1/2 are the real parameters that you use (they are numbers chosen by you). If you find something wrong, modify your code, save it, and type 'imp.reload(T)', then your code gets updated. To do so, you need to type 'import imp' at first.
	# IDLE is a good tool of editting python code, but I recommend you to execute the code in the terminal and debug it with python prompt (not the IDLE prompt). Go to the repository under which 'python.exe' is placed, double click the python icon or type 'python.exe' in the terminal to run the python prompt, put your code in the same repository, then you can import it into the prompt.
    Y1, M1, D1, Y2, M2, D2 = eval(input('Input Y1, M1, D1, Y2, M2, D2:\n'))
    out = day(Y1,M1,D1,Y2,M2,D2)
    print('The interval is '+str(out[0])+' day(s).')
    print('Execution time: '+str(out[1])+' s.')
