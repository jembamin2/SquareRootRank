# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:02:38 2024

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

def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        matrix = []
        r,c=map(int,fin.readline().split())
        for i in range (r):
            tmp = fin.readline().split()
            matrix.append(list(map(float, tmp)))
    return np.array(matrix)

def calculate_rank(matrix, tol=1e-14):
   
    u, s, vh = np.linalg.svd(matrix)
    return np.sum(s > tol)

def calculate_singular(matrix, tol=1e-14):
   
    u, s, vh = np.linalg.svd(matrix)
    singular_vectors=[singular for singular in s if singular>tol]
    
    return min(singular_vectors)

def apply_mask(sqrt_matrix, mask):
    
    return sqrt_matrix * mask

def greedy_randomized_construction(sqrt_matrix, alpha):
    """
    Construction gloutonne randomisée pour générer un masque initial.
    
    Args:
        sqrt_matrix (numpy.ndarray): Matrice après racine carrée élémentaire.
        alpha (float): Paramètre pour contrôler le degré de randomisation.
        
    Returns:
        numpy.ndarray: Masque généré.
    """
    n, m = sqrt_matrix.shape
    mask = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            # Test des deux options (+1 et -1)
            ranks = []
            for sign in [1, -1]:
                temp_mask = mask.copy()
                temp_mask[i, j] = sign
                transformed_matrix = apply_mask(sqrt_matrix, temp_mask)
                ranks.append(calculate_rank(transformed_matrix))
            
            # Liste restreinte de candidats (RCL)
            min_rank = min(ranks)
            max_rank = max(ranks)
            threshold = min_rank + alpha * (max_rank - min_rank)
            rcl = [sign for sign, rank in zip([1, -1], ranks) if rank <= threshold]
            
            # Choisir un signe aléatoire dans la RCL
            chosen_sign = random.choice(rcl)
            mask[i, j] = chosen_sign
    
    return mask
'''
def greedy_randomized_construction(sqrt_matrix, alpha):
    """
    Construction gloutonne randomisée pour générer un masque initial,
    prenant en compte le rang et la plus petite valeur singulière.
    
    Args:
        sqrt_matrix (numpy.ndarray): Matrice après racine carrée élémentaire.
        alpha (float): Paramètre pour contrôler le degré de randomisation.
        
    Returns:
        numpy.ndarray: Masque généré.
    """
    n, m = sqrt_matrix.shape
    mask = np.ones((n, m))  # Initialisation du masque à 1 partout
    
    for i in range(n):
        for j in range(m):
            # Tester les deux options possibles (+1 ou -1)
            ranks = []
            singulars = []
            
            for sign in [1, -1]:
                temp_mask = mask.copy()
                temp_mask[i, j] = sign
                transformed_matrix = apply_mask(sqrt_matrix, temp_mask)
                
                # Calculer le rang et la plus petite valeur singulière
                rank = calculate_rank(transformed_matrix)
                singular = calculate_singular(transformed_matrix)
                
                ranks.append(rank)
                singulars.append(singular)
            
            # Liste restreinte de candidats (RCL)
            min_rank = min(ranks)
            max_rank = max(ranks)
            min_singular = min(singulars)
            max_singular = max(singulars)
            
            # Calcul des seuils pour les deux critères
            rank_threshold = min_rank + alpha * (max_rank - min_rank)
            singular_threshold = min_singular + alpha * (max_singular - min_singular)
            
            # RCL : options satisfaisant les deux critères
            rcl = [
                sign
                for sign, rank, singular in zip([1, -1], ranks, singulars)
                if rank <= rank_threshold and singular >= singular_threshold
            ]
            
            # Choisir un signe aléatoire parmi les options de la RCL
            chosen_sign = random.choice(rcl)
            mask[i, j] = chosen_sign  # Appliquer le signe choisi
    
    return mask
'''
def local_search(mask, sqrt_matrix):
    
    n, m = mask.shape
    best_mask = mask.copy()
    matrix = apply_mask(sqrt_matrix, best_mask)
    best_rank = calculate_rank(matrix)
    smallest_singular = calculate_singular(matrix)
    
    for _ in range(n * m):  # Nombre d'itérations limité pour la recherche locale
        i, j = random.randint(0, n - 1), random.randint(0, m - 1)
        new_mask = best_mask.copy()
        new_mask[i, j] *= -1  # Swap
        
        new_matrix = apply_mask(sqrt_matrix, new_mask)
        new_rank = calculate_rank(new_matrix)
        singular = calculate_singular(new_matrix)
        
        if new_rank < best_rank: # Stocker le meilleur mask
            best_mask = new_mask
            best_rank = new_rank
            smallest_singular = singular
            
            
        elif new_rank == best_rank and singular<smallest_singular:
            best_mask = new_mask
            smallest_singular = singular
    
    return best_mask


def grasp_minimize_sqrt_rank(A, max_iterations=50, alpha=0.3, tol=1e-14):
    """
    Algorithme GRASP pour minimiser le square root rank d'une matrice.
    
    Args:
        A (numpy.ndarray): Matrice d'entrée.
        max_iterations (int): Nombre maximum d'itérations.
        alpha (float): Paramètre pour contrôler le degré de randomisation.
        tol (float): Tolérance pour le calcul du rang.
        
    Returns:
        tuple: Meilleur masque trouvé et le rang minimal atteint.
    """
    sqrt_matrix = np.sqrt(np.abs(A))  # Racine carrée élément par élément
    best_mask = np.ones_like(A)
    best_rank = calculate_rank(sqrt_matrix, tol)
    smallest_singular = calculate_singular(sqrt_matrix, tol)
    
    for _ in tqdm(range(max_iterations)):
        # Étape 1 : Construction gloutonne randomisée
        mask = greedy_randomized_construction(sqrt_matrix, alpha)
        
        # Étape 2 : Recherche locale
        mask = local_search(mask, sqrt_matrix)
        
        # Mise à jour de la meilleure solution
        transformed_matrix = apply_mask(sqrt_matrix, mask)
        current_rank = calculate_rank(transformed_matrix, tol)
        current_singular = calculate_singular(transformed_matrix, tol)
        
        if current_rank < best_rank:
            best_mask = mask
            best_rank = current_rank
            smallest_singular = current_singular
        
        elif current_rank == best_rank and current_singular < smallest_singular:
            best_mask = mask
            smallest_singular = current_singular
    
    return best_mask, best_rank, smallest_singular

#%%
matrix=read_matrix("synthetic_matrice.txt")
matrix=matrices1_ledm(10)

max_iterations = 5000
alpha = 0.1
tol = 1e-10 

best_mask, best_rank, smallest_singular = grasp_minimize_sqrt_rank(matrix, max_iterations, alpha, tol)

print("Meilleur masque trouvé :")
print(best_mask)
print(f"Rang minimal atteint : {best_rank}")
print(f"Plus petite valeur singulière : {smallest_singular}")
