# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:51:18 2024

@author: benja
"""

import numpy as np
import random
from tqdm import tqdm
#%%

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

from scipy.linalg import circulant
def matrices2_slackngon(n):
  M  = circulant(np.cos(np.pi/n)-np.cos(np.pi/n + 2*np.pi*np.arange(0,n,1)/n))
  M /= M[0,2]
  M  = np.maximum(M,0)
  for i in range(n):
    M[i,i] = 0
    if i<n-1:
      M[i,i+1] = 0
    else:
      M[i,0] = 0
  return M

def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        matrix = []
        r,c=map(int,fin.readline().split())
        for i in range (r):
            tmp = fin.readline().split()
            matrix.append(list(map(float, tmp)))
    return np.array(matrix)

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]          


def local_search(matrix,T,nbr_swap):
    
    n, m = matrix.shape
    best_mask = np.ones(matrix.shape)
    best_rank,smallest_singular = fobj(matrix,best_mask)
    
    counter=0
    for _ in tqdm(range(T)):  # Nombre d'itérations limité pour la recherche locale
    
    
        for k in range(nbr_swap):
            
            i, j = random.randint(0, n - 1), random.randint(0, m - 1)
            new_mask = best_mask.copy()
            new_mask[i, j] *= -1  # Swap
        
            
        new_rank,singular=fobj(matrix,new_mask)
        
        
        if new_rank < best_rank: # Stocker le meilleur mask
            best_mask = new_mask
            best_rank = new_rank
            smallest_singular = singular
            nbr_swap=1
            
            
        elif new_rank == best_rank and singular<smallest_singular:
            best_mask = new_mask
            smallest_singular = singular
            nbr_swap=1
        
        else : 
            nbr_swap+=1
        
        if nbr_swap>1000:
            break
        
    return best_mask,best_rank,smallest_singular


#%%

matrix=read_matrix("correl5_matrice.txt")
matrix=matrices1_ledm(120)
#matrix=matrices2_slackngon(50)

a,b,c=local_search(matrix, 100000,1)

print(f"best rank = {b}")
print(f"smallest singular = {c}")

