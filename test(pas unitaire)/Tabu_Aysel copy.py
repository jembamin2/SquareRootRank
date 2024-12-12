import matplotlib.pyplot as plt
import numpy as np
import time


#%%

from scipy.linalg import circulant

def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        lines = fin.readlines()
        n, m = map(int, lines[0].strip().split())
        matrix = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                matrix.append(list(map(float, line.strip().split())))
        if len(matrix) != n or any(len(row) != m for row in matrix):
            raise ValueError("Matrix dimensions do not match the provided data.")
    return np.array(matrix)

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

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

# Appliquer le masque à la matrice
def apply_mask(matrix, mask):
    return matrix * mask

# Calculer le rang
def compute_rank(matrix):
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    rank = np.sum(S > 1e-10)
    return rank

def fobj(M,P=1):
    sing_values = np.linalg.svd(M*P, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def initialize_mask_from_low_rank_blocks(matrix, block_size=10):
    blocks = split_into_blocks(matrix, block_size)
    mask = np.ones(matrix.shape)

    for block, i, j in blocks:
        if np.linalg.matrix_rank(block) <= 2:  # Seuil pour les blocs à faible rang  Il faudrait print pour savoir ce que ca retourne car definition dit tol=None
            mask[i:i+block_size, j:j+block_size] = -1

    return mask

def split_into_blocks(matrix, block_size):
    """
    Divise une matrice en blocs de taille `block_size x block_size`.
    """
    n, m = matrix.shape
    blocks = []
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            blocks.append((block, i, j))  # Garder les indices de début
    return blocks

def optimize_block_mask(block, threshold=1e-12):
    """
    Optimise le masque pour une sous-matrice donnée.
    """
    U, S, Vh = np.linalg.svd(block, full_matrices=False)
    mask = np.ones(block.shape)
    for i, singular_value in enumerate(S):
        if singular_value < threshold:
            mask[i, :] = -1  # Marquer les lignes faibles
    return mask

def reconstruct_global_mask(blocks, masks, matrix_shape):
    """
    Reconstruit le masque global à partir des masques des blocs.
    """
    global_mask = np.ones(matrix_shape)
    for (block, i, j), mask in zip(blocks, masks):
        n, m = mask.shape
        global_mask[i:i+n, j:j+m] = mask
    return global_mask

def initialize_mask_from_blocks(matrix, block_size, threshold=1e-12):
    # Diviser la matrice en blocs
    blocks = split_into_blocks(matrix, block_size)
    # print(blocks)
    
    # Optimiser le masque pour chaque bloc
    masks = [optimize_block_mask(block, threshold) for block, _, _ in blocks]
    
    # Reconstruire le masque global
    global_mask = reconstruct_global_mask(blocks, masks, matrix.shape)
    return global_mask

def generate_neighbors(current_mask, num_neighbors, num_modifications):
    """
    Generates neighbors by flipping a random set of entries in the current_mask.
    """
    n_rows, n_cols = current_mask.shape
    neighbors = []

    # Randomly pick `num_modifications` positions to flip
    total_elements = n_rows * n_cols
    all_indices = np.arange(total_elements)
    
    for _ in range(num_neighbors):
        # Create a new neighbor by copying the current mask
        neighbor = current_mask.copy()
        
        # Randomly select `num_modifications` indices to flip
        flip_indices = np.random.choice(all_indices, num_modifications, replace=False)
        
        # Flip the selected indices (invert their values)
        for idx in flip_indices:
            i, j = divmod(idx, n_cols)  # Convert flat index to 2D (i, j)
            neighbor[i, j] *= -1
        
        neighbors.append(neighbor)
    
    return neighbors



def tabu_search_with_plot(matrix, initial_mask, tot_resets, num_n, num_m, tabu_size, max_no_improve, max_iterations):
    """
    Recherche tabou pour minimiser le rang avec graphiques.
    """
    # Historique pour tous les voisinages
    all_rank_histories = []
    rank_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    best_overall_mask = initial_mask.copy()
    total_resets = 0

    best_rank = best_overall_rank;
    
    tabu_set = set()

    
    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = initial_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(matrix)
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0

        rank_history = []
        print(f"=== Exploration du voisinage {total_resets + 1} ===")
        for iteration in range(max_iterations):
            num_neighbors, num_modifications = dynamic_neigh_modulation(iteration, max_iterations, num_n, num_m)
            neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)            # Générer plusieurs voisins
            # neighbors = []
            # n_rows, n_cols = current_mask.shape
            # num_neighbors = num_n
            # num_modifications = num_m
            
            if iteration == 1:
                print(f"neigh = {num_neighbors}")
                print(f"mod = {num_modifications}")
                # Générer les voisins dynamiquement
            # neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)

        
            # Évaluer tous les voisins
            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    masked_matrix = apply_mask(matrix, neighbor)
                    neighbor_rank, neighbor_singular_value = fobj(masked_matrix)
                    evaluated_neighbors.append((neighbor_rank, neighbor_singular_value, neighbor_bytes))
            
            # Sort the neighbors by rank, then by singular value
            evaluated_neighbors.sort(key=lambda x: (x[0], x[1]))
            
            # Prendre le meilleur voisin (ou plusieurs)
            if evaluated_neighbors:
                best_neighbor_rank, best_neighbor_singular_value, best_neighbor_bytes = evaluated_neighbors[0]
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape) #je comrpends pas TODO
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value
        
                rank_history.append((best_rank, current_singular_value))

                if compareP1betterthanP2(matrix,current_mask,best_mask):
                    best_mask = current_mask.copy()
                    best_rank = current_rank
                    best_singular_value = current_singular_value
                    print(f"no improve: {iteration - no_improve}, rank = {current_rank}")
                    no_improve = 0
                else:
                    no_improve += 1

                # Add to tabu set
                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()
        
            if best_rank == 2:
                break
            
            # Si aucune amélioration après plusieurs itérations
            if no_improve >= max_no_improve:
                print(f"num iterations: {iteration}, rank = {best_rank}")
                break

        
            #print(f"Iteration {iteration + 1}: Current Rank = {current_rank}, Best Rank = {best_rank}")

        # Mettre à jour le meilleur résultat global
        if compareP1betterthanP2(matrix,best_mask,best_overall_mask):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask


        # Sauvegarder l'historique du voisinage
        all_rank_histories.append(rank_history)

        # Réinitialiser pour un nouveau voisinage
        total_resets += 1

    # Tracer les rangs pour tous les voisinages
    plt.figure(figsize=(12, 8))

    # Parcourir chaque voisinage et tracer son historique de rangs
    for i, rank_history in enumerate(all_rank_histories):
        iterations = range(len(rank_history))  # Réinitialiser les itérations pour chaque voisinage
        ranks = [rank for rank, _ in rank_history]  # Extraire les rangs
        plt.plot(iterations, ranks, label=f"Voisinage {i + 1}")  # Tracer une ligne pour chaque voisinage
    
    # Ajouter une ligne pour le meilleur rang global
    plt.axhline(y=best_overall_rank, color='r', linestyle='--', label="Meilleur rang global")
    
    # Ajouter des détails pour rendre le graphique plus lisible
    plt.xlabel("Itérations (réinitialisées pour chaque voisinage)")
    plt.ylabel("Rang")
    plt.title("Évolution du rang pour chaque voisinage (Recherche Tabou)")
    plt.legend()  # Ajouter une légende
    plt.grid(True)  # Afficher la grille
    plt.show()

    return best_overall_mask, best_overall_rank


def reconstruct_mask_from_blocks(matrix, blocks, max_rank=3):
    """
    Reconstructs the global mask by combining the masks of individual blocks.
    """
    n, m = matrix.shape
    global_mask = np.ones((n, m))  # Initialize the mask with 1 (valid blocks)

    for block, i, j, h, w in blocks:
        # Mark the block as low-rank if its rank is <= max_rank
        if np.linalg.matrix_rank(block) <= max_rank:
            global_mask[i:i+h, j:j+w] = -1  # Set low-rank blocks to -1
    
    return global_mask



def initialize_mask_from_blocks_fast(matrix, block_size, mask_method, **kwargs):
    """
    Initialise rapidement un masque global en appliquant un masque local à chaque sous-bloc de la matrice.
    
    Args:
        matrix (np.ndarray): La matrice d'origine.
        block_size (int): Taille des blocs carrés.
        mask_method (callable): Méthode utilisée pour générer un sous-masque pour un bloc.
        **kwargs: Paramètres supplémentaires pour la méthode de génération de masque.
    
    Returns:
        np.ndarray: Le masque global appliqué à la matrice.
    """
    n_rows, n_cols = matrix.shape
    mask = np.ones_like(matrix)  # Masque global initialisé à 1
    
    # Découper et traiter les blocs
    for i in range(0, n_rows, block_size):
        for j in range(0, n_cols, block_size):
            # Extraire le sous-bloc
            block = matrix[i:i+block_size, j:j+block_size]
            
            # Générer un sous-masque rapide pour ce bloc
            block_mask = simple_block_mask(block, **kwargs)
            
            # Appliquer le sous-masque sur la matrice globale
            mask[i:i+block_size, j:j+block_size] = block_mask
    
    return mask

def simple_block_mask(block, threshold=1e-12):
    """
    Crée un masque pour un bloc en marquant les valeurs proches de zéro.
    
    Args:
        block (np.ndarray): Le bloc à analyser.
        threshold (float): Seuil pour définir les valeurs faibles.
    
    Returns:
        np.ndarray: Masque du bloc (-1 pour les valeurs faibles, 1 sinon).
    """
    mask = np.ones_like(block)
    mask[np.abs(block) < threshold] = -1  # Masquer les valeurs faibles
    return mask

def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    """
    Dynamically adjusts the number of neighbors and modifications based on the iteration number.
    """
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))  # Reduce neighbors as progress increases
    num_modifications = max(1, int(initial_modifications * (1 - progress)))  # Reduce modifications, ensure at least 1
    return num_neighbors, num_modifications


#%%
# Exemple d'application
if __name__ == "__main__":
    #original_matrix = read_matrix("synthetic_matrice.txt")
    # original_matrix = read_matrix("correl5_matrice.txt")
    original_matrix = matrices1_ledm(120)
    sqrt_matrix = np.sqrt(original_matrix)
    initial_mask = np.ones_like(sqrt_matrix)
    in_rank, s = fobj(sqrt_matrix)
    print("Matrice originale :\n", original_matrix)
    print("Initial Rank :\n", in_rank)
    print("Initial Sing Values :\n", s)
    print("Matrice sq rt :\n", sqrt_matrix)

    # Exemple d'application
    block_size = 6  # Taille des sous-matrices
    threshold = 1e-12  # Seuil pour détecter les contributions faibles

    #block_mask = initialize_mask_from_low_rank_blocks(sqrt_matrix, block_size=5)
    block_mask = initialize_mask_from_blocks(original_matrix, block_size)
    print("Masque basé sur blocs :", block_mask)



    # Appliquer la recherche tabou avec graphique
    start = time.time()
    #best_mask, best_rank = tabu_search_with_plot(sqrt_matrix, initial_mask,mod_neighbors = (0.5, 0.01), mod_mod = (0.025, 0.01), tabu_size=1000, max_no_improve=2000, max_iterations=1000000)
    best_mask, best_rank = tabu_search_with_plot(
        sqrt_matrix, block_mask,
        tot_resets = 15,
        num_n = 60,
        num_m = 1,
        tabu_size=10000, 
        max_no_improve=5000, 
        max_iterations=10000000)
    stop = time.time()
    print((stop-start)//60, (stop-start)%60)
    
    # Résultats finaux
    print("\nMeilleur masque trouvé :\n", best_mask)

    print("Rang minimal obtenu :", best_rank)

    # Appliquer le meilleur masque trouvé
    final_matrix = apply_mask(sqrt_matrix, best_mask)
