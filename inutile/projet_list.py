import numpy as np
import math

#%%
def read_matrix(input):
    with open(input,'r') as fin:
        matrix = []
        while True:
            tmp=fin.readline().rsplit(",")
            if tmp==[""]:
                break
            matrix.append(list(map(int,tmp)))
    return matrix


def setup_mask(length):
    mask = [[]]*length
    for line in mask:
        line.append(1)
    return mask

def setup_sqrt_matrix(matrix):
    sqrt_matrix=[]
    for line in matrix:
        sqrt_matrix.append([math.sqrt(item) for item in line])
    return sqrt_matrix

#%%
matrix = read_matrix("input.txt")

a=len(matrix)

mask = setup_mask(a)

sqrt_matrix=setup_sqrt_matrix(matrix)



