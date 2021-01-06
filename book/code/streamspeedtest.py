import numpy as np
import pandas
from numpy.random import rand

def generate_arrays(N):
    rand(N).tofile('/tmp/x.data', sep='\n')
    rand(N).tofile('/tmp/y.data', sep='\n')


def myfun(xfile, yfile):
    fx = open(xfile, 'r')
    fy = open(yfile, 'r')

    retval = 0.0
    for x in fx:
        y = fy.next()
        retval += float(x) * float(y)

    fx.close()
    fy.close()

    return retval


def myfun_pandas(xfile, yfile):
    x = pandas.read_csv(xfile, header=None)
    y = pandas.read_csv(yfile, header=None)

    retval = x.T.dot(y).values
    
    return retval
