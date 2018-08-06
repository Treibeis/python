t0=0;t1=0;t2=0
while t0==0:
    if t1==0:
        n = eval(input ("Please input the number of lines, n:"))
        if n < 1:
            print ("Input error: n should be no less than one")
            continue
    t1=1
    if t2 ==0:
        m = eval(input ("Please input the number of spacings, m:"))
        if m < 0:
            print ("Input error: m should no less than zero")
            continue
    t2 =1
    if 3*n+m-1 > 80:
        print ("3n+m-1 should not be larger than 80")
        t1 =0
        t2 =0
    else:
        t0=1
i=0;j=0
while i<n:
    if j<n-i-1:
        print (" "*(n-i-1),end='')
    if j<2*i+1:
        print ("*"*(2*i+1),end='')
    if j<m:
        print (" "*m,end='')
    if j<2*n-2*i-1:
        print ("*"*(2*n-2*i-1))
    i=i+1
    
    
    

