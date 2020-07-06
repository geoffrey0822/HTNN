import operator as op
from functools import reduce
import math

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def ncr2(n):
    output = 0
    for i in range(1, n):
        x = math.log(math.pow(2, i), 2)
        output += x
    return output
