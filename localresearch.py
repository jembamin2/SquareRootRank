# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:51:18 2024

@author: benja
"""

import numpy as np
import random
from tqdm import tqdm
#%%
def generate_low_rank_matrix(m, n, rank):
    """
    Generates a low-rank matrix by multiplying two random matrices.
    
    Parameters:
        m (int): Number of rows of the matrix.
        n (int): Number of columns of the matrix.
        rank (int): Desired rank of the matrix.
        
    Returns:
        np.ndarray: Low-rank matrix of size (m, n).
    """
    U = np.random.rand(m, rank)  # m x rank matrix
    V = np.random.rand(rank, n)  # rank x n matrix
    low_rank_matrix = np.dot(U, V)  # Matrix multiplication gives an m x n matrix
    return low_rank_matrix

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


def local_search(matrix,T,nbr_swap,neighborhood_exploration):
    
    n, m = matrix.shape
    best_mask = np.ones(matrix.shape)
    best_rank,smallest_singular = fobj(matrix,best_mask)
    
    new_best_mask = best_mask
    new_best_rank,new_smallest_singular = best_rank,smallest_singular
    
    counter=0
    for _ in range(T):  # Nombre d'itérations limité pour la recherche locale
    
        for l in range(int(m*n*neighborhood_exploration)):
            for k in range(nbr_swap):
                
                i, j = random.randint(0, n - 1), random.randint(0, m - 1)
                new_mask = new_best_mask.copy()
                new_mask[i, j] *= -1  # Swap
                
            
            new_rank,singular=fobj(matrix,new_mask)
                
            
            
            if new_rank < new_best_rank: # Stocker le meilleur mask
                new_best_mask = new_mask
                new_best_rank = new_rank
                new_smallest_singular = singular
                #nbr_swap=1
                
                
            elif new_rank == new_best_rank and new_smallest_singular<smallest_singular:
                new_best_mask = new_mask
                new_smallest_singular = singular
                #nbr_swap=1
        
        if new_best_rank < best_rank: # Stocker le meilleur mask
            best_mask = new_best_mask
            best_rank = new_best_rank
            smallest_singular = new_smallest_singular
            #nbr_swap=1
            
            
        elif new_best_rank == best_rank and new_smallest_singular<smallest_singular:
            best_mask = new_best_mask
            smallest_singular = new_smallest_singular
            #nbr_swap=1
        
        # else : 
        #     nbr_swap+=1
        
        # if nbr_swap>3:
        #     nbr_swap=1
            
        
    return best_mask,best_rank,smallest_singular

def hierarchical_block_tabu_heuristic(matrix, initial_block_size=8, scaling_factor=2):
    n = matrix.shape[0]  # Size of the full matrix

    # Initialize a global mask with all ones
    global_mask = np.ones_like(matrix, dtype=int)

    # Start with the smallest block size and progressively increase
    block_size = initial_block_size
    while block_size <= n:
        num_blocks = int(n // block_size)  # Number of blocks along each dimension
        # Refine the mask using the current block size
        for i in tqdm(range(num_blocks)):
            for j in range(num_blocks):
                # Extract the block
                block = matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                
                # Extract the corresponding portion of the global mask
                block_mask_initial = global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                
                # Apply tabu search to optimize the block
                block_mask, _,poubelle = local_search(block,1000,1,1/10*block_size)
                
                # Update the corresponding portion of the global mask
                global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_mask

        # Increase the block size for the next iteration
        block_size *= scaling_factor

    return global_mask

#%%

matrix=read_matrix("correl5_matrice.txt")
matrix=matrices1_ledm(120)
#matrix=matrices2_slackngon(50)


a=hierarchical_block_tabu_heuristic(matrix)

rank,singular=fobj(matrix,a)

print(f"best rank = {rank}")
print(f"smallest singular = {singular}")

