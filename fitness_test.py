import numpy as np
import sys
from PIL import Image

temp=Image.open('Guernica.png')
temp=temp.convert('1')      # Convert to black&white
A = np.array(temp)             # Creates an array, white pixels==True and black pixels==False
a = np.empty((A.shape[0],A.shape[1]),None)    #New array with same size as A
for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j]==True:
            a[i][j]=0
        else:
            a[i][j]=1
            
np.set_printoptions(threshold=sys.maxsize)

a = np.reshape(a, 8000)
a = np.array(a, dtype='int')
b = np.random.choice([0, 1], 8000)

c = (np.absolute(a-b))
print(a)
print(b)
print(c)
print(np.sum(c, axis=0))
print((np.sum(c, axis=0)) / 8000)

def evalComp():
    compare = np.absolute(a-b)
    compare = np.sum(compare, axis=0)
    return compare

print(evalComp())