import numpy as np
import math
import random
import time

from numpy.linalg import svd


def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        matrix = []
        r,c=map(int,fin.readline().split())
        for i in range (r):
            tmp = fin.readline().split()
            matrix.append(list(map(float, tmp)))
    return np.array(matrix)


def setup_mask(shape):
    return np.ones(shape)


def setup_sqrt_matrix(matrix):
    return np.sqrt(matrix)


def swap(mask, shape):
    # Choisir une position aléatoire (ligne, colonne)
    row = random.randint(0, shape[0] - 1)
    col = random.randint(0, shape[1] - 1)
    # Inverser le signe de l'élément correspondant
    mask[row, col] = -mask[row, col]
    return mask


def evaluate_matrix(matrix, tol=1e-10):
    # Décomposer la matrice en valeurs singulières
    U, singular_values, V = svd(matrix)
    # Calculer le rang (nombre de valeurs singulières > tolérance)
    rank = np.sum(singular_values > tol)
    # Obtenir la plus petite valeur singulière non nulle
    smallest_singular = singular_values[-1]
    # Filtrer toutes les valeurs singulières supérieures à la tolérance
    significant_singular_values = singular_values[singular_values > tol]
    test=singular_values
    return rank, smallest_singular, significant_singular_values, U, V,test


def optimize_matrix(sqrt_matrix, mask, num_iterations):
    best_mask = mask.copy()
    best_rank = float('inf')  # Initialiser avec un rang très élevé
    best_singular = float('inf')  # Initialiser avec une valeur singulière très élevée
    best_significant_singular_values = []  # Liste pour stocker les meilleures valeurs singulières

    for iteration in range(num_iterations):
        # Générer un masque aléatoire en inversant un élément au hasard
        new_mask = swap(mask.copy(), mask.shape)
        
        # Appliquer le masque à la matrice
        test_matrix = sqrt_matrix * new_mask
        
        # Calculer le rang et les valeurs singulières significatives
        rank, smallest_singular, significant_singular_values, U, V,test = evaluate_matrix(test_matrix)
        
        # Mettre à jour les meilleurs résultats si nécessaire
        if (rank < best_rank) or (rank == best_rank and smallest_singular < best_singular):
            best_mask = new_mask.copy()
            best_rank = rank
            best_singular = smallest_singular
            best_significant_singular_values = significant_singular_values
            best_U, best_V = U, V
            btest = test
        # Affichage des résultats intermédiaires
        print(f"{iteration}: rank={rank} smallest_singular={smallest_singular}")

    # Retourner les meilleurs résultats trouvés
    return best_mask, best_rank, best_singular, best_significant_singular_values, best_U, best_V,btest


def filter_and_reform_matrix(matrix, U, S, V, tol=1e-10):
    # Créer une matrice Sigma_filtered avec les mêmes dimensions que matrix, explicitement en float
    Sigma_filtered = np.zeros_like(matrix, dtype=float)
    
    # Remplir la diagonale de Sigma_filtered avec les valeurs singulières significatives
    for i, singular_value in enumerate(S):
        Sigma_filtered[i, i] = singular_value
    
    # Reformuler la matrice avec U, Sigma_filtered, et Vt
    reformed_matrix = U @ Sigma_filtered @ V
    
    return Sigma_filtered, reformed_matrix

def validate_solution(original_matrix, reformed_matrix):
    # Utiliser la norme Frobenius pour comparer directement
    return np.linalg.norm(original_matrix - reformed_matrix**2, ord='fro')

#%% 
start = time.time()

matrix = read_matrix("input.txt")
shape = matrix.shape

mask = setup_mask(shape)
sqrt_matrix = setup_sqrt_matrix(matrix)

num_iterations = 1000000
best_mask, best_rank, best_singular, best_significant_singular_values, best_U, best_V,btest = optimize_matrix(sqrt_matrix, mask, num_iterations)

print("Best rank:", best_rank)
print("Smallest singular value:", best_significant_singular_values[-1])

# Reformuler la matrice et valider la solution
S_filtered,reformed_matrix = filter_and_reform_matrix(matrix,best_U, best_significant_singular_values, best_V)
difference = validate_solution(matrix, reformed_matrix)

print("Difference between original and reformed matrix:", difference)
if difference < 1e-5:
    print("Solution is valid.")
else:
    print("Solution is not valid.")

print("Execution time:", time.time() - start)
