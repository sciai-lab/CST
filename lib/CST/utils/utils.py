def factor_number(x,reversed=False):
    i=1
    if x in [3,5,7,11,13,17,19,23]:
        x+=1
    suma=x+1
    
    while i<=x**0.5:
        if x%i==0:
            if x//i+i<=suma:
                k1,k2=x//i,i
        i+=1
    if reversed:
        return k1, k2
    return k2,k1

