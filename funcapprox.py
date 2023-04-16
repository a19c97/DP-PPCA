from numpy import sign
def f(lor,b):
    su=0
    for x in lor:
        su+=1/(b-x)
    return su


def findz(lor, orerr, debug=False):
    lor.sort()
    for i in range(len(lor)-1):
        err=min(orerr,abs(lor[i]-lor[i+1])/100)
        if debug:
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
            if debug:
                print(pr,mid,nx, lm,fm,rm)
            if(sign(lm)!=sign(fm)):
                nx=mid
            else:
                pr=mid
            mid=(pr+nx)/2
            fm=f(lor,mid)


        return mid

if __name__ == '__main__':
    lor=[-4,-5,-7,-10]
    b = findz(lor, 0.0001)
    print('optimal b: ' + str(b))
    print('f value: ' + str(f(lor, b)))
