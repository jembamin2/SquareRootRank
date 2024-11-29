import numpy as np
import math
import random
import time
from tqdm import tqdm


from numpy.linalg import svd

def LEDM (n,m):
    M=np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
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


def setup_mask(shape):
    return np.ones(shape)


def setup_sqrt_matrix(matrix):
    return np.sqrt(matrix)


def swap(mask, shape, num_swap):
    
    for i in range(num_swap):
        # Choisir une position aléatoire (ligne, colonne)
        row1 = random.randint(0, shape[0] - 1)
        col1 = random.randint(0, shape[1] - 1)
    
  
        # Inverser le signe de l'élément correspondant
        mask[row1, col1] = -mask[row1, col1]
        
    return mask


def evaluate_matrix(matrix, tol=1e-14):
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
    last_mask = mask.copy()
    last_rank = float('inf')
    last_singular = float('inf')  
    last_significant_singular_values = []  
    
    
    for iteration in tqdm(range(num_iterations)):
        # Générer un masque aléatoire en inversant un élément au hasard
        
        random_swap = random.randint(1, 3)
        random_recuit_swap = random.uniform(0, 1)
        random_recuit = random.uniform(0, 1)
        
        num_swap=1
        if random_recuit_swap<0.05:
            num_swap=random_swap
        
        new_mask = swap(last_mask, mask.shape,num_swap)
        
        # Appliquer le masque à la matrice
        test_matrix = sqrt_matrix * new_mask
        
        # Calculer le rang et les valeurs singulières significatives
        rank, smallest_singular, significant_singular_values, U, V,test = evaluate_matrix(test_matrix)
        
        if (rank < last_rank) or (rank == last_rank and significant_singular_values[-1] < last_significant_singular_values[-1]):
            last_mask = new_mask.copy()
            last_rank = rank
            last_significant_singular_values = significant_singular_values
            
        
        # Mettre à jour les meilleurs résultats si nécessaire
        if (rank < best_rank) or (rank == best_rank and significant_singular_values[-1] < best_significant_singular_values[-1]):
            best_mask = new_mask.copy()
            best_rank = rank
            best_singular = smallest_singular
            best_significant_singular_values = significant_singular_values
            last_mask = new_mask.copy()
            last_rank = rank
            last_significant_singular_values = significant_singular_values
            best_U, best_V = U, V
            btest = test
            continue
        
        
        # if random_recuit<0.07:
        #     best_mask=-best_mask #si j'inverse tout le masque
        
        if random_recuit<0.05:
            last_mask=swap(best_mask, mask.shape,int(len(best_mask) * 0.25))
        
            
            
            
            
        # Affichage des résultats intermédiaires
        # print(f"{iteration}: rank={best_rank} smallest_singular={best_significant_singular_values[-1]}")
        #print(f"{iteration}: rank={rank} smallest_singular={significant_singular_values[-1]}")

    # Retourner les meilleurs résultats trouvés
    return best_mask, best_rank, best_singular, best_significant_singular_values, best_U, best_V,btest


def filter_and_reform_matrix(matrix, U, S, V, tol=1e-6):
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


# matrix = read_matrix("input.txt")

matrix = LEDM(7, 7)

# matrix = (np.random.rand(7,3)*10)@(np.random.rand(3,7)*10)
# matrix = matrix**2

shape = matrix.shape

mask = setup_mask(shape)
sqrt_matrix = setup_sqrt_matrix(matrix)
num_iterations = 100000
best_mask, best_rank, best_singular, best_significant_singular_values, best_U, best_V,btest = optimize_matrix(sqrt_matrix, mask, num_iterations)
print()
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
