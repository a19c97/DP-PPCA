from numpy import sign
def f(lor,b):
    su=0
    for x in lor:
        su+=1/(b-x)
    return su


def findz(lor, err):
    lor.sort()
    err=min(err,min([abs(lor[i]-lor[i+1])/10 for i in range(len(lor)-1)]))
    for i in range(len(lor)-1):
        pr=lor[i]+err
        nx=lor[i+1]-err
        if(sign(f(lor,pr))==sign(f(lor,nx))):
           continue
        mid=((pr+nx)/2)
        fm=f(lor,mid)
        ct=0
        while(abs(fm)>err):
           
            lm=f(lor, pr)
            rm=f(lor, nx)
            print(pr,mid,nx, lm,fm,rm)
            if(sign(lm)!=sign(fm)):
                nx=mid
            else:
                pr=mid
            mid=(pr+nx)/2
            fm=f(lor,mid)


        return mid
           
lor=[-1,-1.2,-1.3,-1.4]
print(findz(lor,0.001))
