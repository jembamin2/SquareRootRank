import numpy as np
import math
import random

#%%
def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        while True:
            tmp=fin.readline().strip().split(",")
            if tmp==[""]:
                break
            matrix.append(list(map(int,tmp)))
    return np.array(matrix)


def setup_mask(shape):
    mask=np.ones(shape)
    return mask

def setup_sqrt_matrix(matrix):
    sqrt_matrix=np.zeros(matrix.shape)
    sqrt_matrix=np.sqrt(matrix)
    return sqrt_matrix

def swap(mask,index,shape):
    mask[index%shape[0],index//shape[0]]=-mask[index%shape[0],index//shape[0]]
    return mask

def count_nonzero(array, tol=10**-1):
    count=0
    for item in array:
        if item>tol:
            count+=1
    return count

#%%
matrix = read_matrix("input.txt")

a=matrix.shape

mask = setup_mask(a)

sqrt_matrix=setup_sqrt_matrix(matrix)

#%%
min_rank=5
best_S=np.zeros(a)
best_mask=np.zeros(a)
for _ in range(2000):
    mask=swap(mask,random.randint(0,mask.shape[0]**2-1),a)
    S=np.linalg.svd(sqrt_matrix*mask)[1]
    rank=count_nonzero(S)
    if rank<min_rank:
        min_rank=rank
        best_S=S
        best_mask=mask

U,S,Vh=np.linalg.svd(sqrt_matrix*mask)
print(((U*S)@Vh)**2)
print(min_rank)
print(best_S)
print(best_mask)



