
def power2_upperbound(a):
    e = 0
    v = 1
    while v < a:
        v *= 2
        e += 1
    return e

def upper_bound(vt, a):
    '''
    first index > a in (sorted) vt
    '''
    l = 0
    r = len(vt)
    while l < r:
        m = (l+r)//2
        if vt[m] <= a:
            l = m+1
        else:
            r = m
    assert(l == r)
    return r

def upper_bound_non_strict(vt, a):
    '''
    first index >= a in (sorted) vt
    '''
    l = 0
    r = len(vt)
    while l < r:
        m = (l+r)//2
        if vt[m] < a:
            l = m+1
        else:
            r = m
    assert(l == r)
    return r