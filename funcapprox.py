from numpy import sign
def f(lor,b):
    su=0
    for x in lor:
        su+=1/(b-x)
    return su


def findz(lor, orerr):
    lor.sort()
    for i in range(len(lor)-1):
        err=min(orerr,abs(lor[i]-lor[i+1])/100)
        print(err)
        pr=lor[i]+err
        nx=lor[i+1]-err
        if(sign(f(lor,pr))==sign(f(lor,nx))):
           print("not supposed to happen")
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
           
lor=[-4,-5,-7,-10]
print(findz(lor,0.001))
