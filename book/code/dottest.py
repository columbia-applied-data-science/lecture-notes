import numpy as np

@profile
def pythondot(list1, list2):
    dotsum = 0
    for i in xrange(len(list1)):
        dotsum += list1[i] * list2[i]
    
    return dotsum

def numpydot(arr1, arr2):
    return arr1.dot(arr2)

def testfuns(arrsize, numiter):
    mylist = [1] * arrsize
    myarray = np.ones(arrsize)

    for i in xrange(numiter):
        temp = pythondot(mylist, mylist)
        temp = numpydot(myarray, myarray)

if __name__ == '__main__':
    testfuns(1000, 10)
