import matplotlib.pyplot as plt
import numpy as np
import time
import opti_combi_projet_pythoncode_texte_v2 as opti
import numpy as np
import random
from tqdm import tqdm
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

def fobj(M):
    sing_values = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]

def fobj2(M, P):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]
def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M*P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M*P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def initialize_mask_from_low_rank_blocks(matrix, block_size=10):
    blocks = split_into_blocks(matrix, block_size)
    mask = np.ones(matrix.shape)

    for block, i, j in blocks:
        if np.linalg.matrix_rank(block) <= 2:  # Seuil pour les blocs à faible rang
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

def optimize_block_mask(block, threshold=1e-2):
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

def initialize_mask_from_blocks(matrix, block_size, threshold=1e-2):
    # Diviser la matrice en blocs
    blocks = split_into_blocks(matrix, block_size)
    
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

#%%

def add_random_masks(existing_masks, num_random, mask_shape):
    for _ in range(num_random):
        random_mask = np.random.choice([-1, 1], size=mask_shape)
        existing_masks.append(random_mask)
    
    return existing_masks

def add_noise_to_mask(mask, noise_rate=0.2):
    noisy_mask = mask.copy()
    total_elements = mask.size
    num_noisy_elements = int(noise_rate * total_elements)

    noisy_indices = np.random.choice(total_elements, num_noisy_elements, replace=False)
    flattened_mask = noisy_mask.flatten()
    flattened_mask[noisy_indices] = -1 * flattened_mask[noisy_indices]
    
    return flattened_mask.reshape(mask.shape)


def hybrid_initialize_population(matrix, population_size):
    block_based = initialize_population_with_blocks(matrix, block_size=1, population_size=population_size)
    gaussian_based = initialize_population_gaussian(matrix, block_size=1, population_size=population_size)
    checker_based = initialize_population_checkerboard(matrix, block_size=1, population_size=population_size)
    strided_based = initialize_population_strided_blocks(matrix, block_size=1, population_size=population_size)

    # Add noise-based masks ADD TO ALL
    noise_masks = [add_noise_to_mask(mask, noise_rate=0.2) for mask in block_based]

    # Select top masks from each strategy based on objective function
    top_block = sorted(block_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_gaussian = sorted(gaussian_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_checker = sorted(checker_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_strided = sorted(strided_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_noise = sorted(noise_masks, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]

    # Combine top masks from all strategies
    initial_population = top_block + top_gaussian + top_checker + top_strided + top_noise

    # Ensure deduplication and maintain final population size
    unique_population = ensure_unique_population(initial_population, population_size, matrix)

    # Print the top masks for each initialization
    print_top_5(deduplicate_masks(top_block), matrix, "Block-based")
    print_top_5(deduplicate_masks(top_gaussian), matrix, "Gaussian-based")
    print_top_5(deduplicate_masks(top_checker), matrix, "Checkerboard-based")
    print_top_5(deduplicate_masks(top_strided), matrix, "Strided-based")
    print_top_5(deduplicate_masks(top_noise), matrix, "Noise-based")
    print_top_5(deduplicate_masks(unique_population), matrix, "Final Initial Population")

    
    return unique_population[:-2] + [initialize_mask_from_blocks(matrix,2)] + [hierarchical_block_tabu_heuristic(matrix)]

def deduplicate_masks(masks):
    unique_masks = set()
    deduplicated = []
    for mask in masks:
        mask_tuple = tuple(map(tuple, mask))  # Convert numpy array to a tuple of tuples
        if mask_tuple not in unique_masks:
            unique_masks.add(mask_tuple)
            deduplicated.append(mask)
    return deduplicated

def ensure_unique_population(masks, population_size, matrix=None, block_size=4):
    """
    Ensures a population of unique masks of the desired size.
    Deduplicates the masks and generates additional masks if necessary.
    """
    unique_masks = set()
    deduplicated = []

    def to_tuple(mask):
        return tuple(map(tuple, mask))  # Convert numpy array to a hashable tuple

    # Deduplicate existing masks
    for mask in masks:
        mask_tuple = to_tuple(mask)
        if mask_tuple not in unique_masks:
            unique_masks.add(mask_tuple)
            deduplicated.append(mask)

    # Dynamically generate additional masks if needed
    while len(deduplicated) < population_size:
        # Generate additional masks using the hybrid strategy
        additional_masks = generate_improved_random_masks(
            [],  # Start with an empty list of masks
            num_random=population_size - len(deduplicated),
            mask_shape=masks[0].shape,
            matrix=matrix,
            block_size=block_size
        )
        for mask in additional_masks:
            mask_tuple = to_tuple(mask)
            if mask_tuple not in unique_masks:
                unique_masks.add(mask_tuple)
                deduplicated.append(mask)
    return deduplicated[:population_size]


def initialize_population_with_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks using block-based strategies.
    Dynamically adjusts block sizes within the function.
    """
    n, m = matrix.shape
    population = []  # Start with an all-ones mask

    # Dynamically generate block sizes by doubling up to the matrix dimensions
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2  # Double the block size for the next iteration

    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m))  # Randomly select a mask from the existing population

            # Calculate the shift for this mask (wrap around the matrix)
            block_offset = population_index * block_size % n, population_index * block_size % m

            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    # Shift the block indices cyclically
                    shifted_i = (i + block_offset[0]) % n
                    shifted_j = (j + block_offset[1]) % m

                    # Create a block view from the shifted indices
                    block = matrix[shifted_i:shifted_i+block_size, shifted_j:shifted_j+block_size]

                    # If the block is low-rank, set corresponding region in the mask to -1
                    if np.linalg.matrix_rank(block) <= 3:
                        mask[shifted_i:shifted_i+block_size, shifted_j:shifted_j+block_size] = -1

            population.append(mask)
    
    return population


def initialize_population_random_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks by randomly selecting blocks and setting them to 1 or -1.
    """
    n, m = matrix.shape
    population = []
    
    for population_index in range(population_size):
        mask = np.ones((n, m))  # Start with all ones

        # Randomly select blocks and set them to -1
        for i in range(0, n, block_size):
            for j in range(0, m, block_size):
                # Randomly decide if this block should be masked
                if np.random.rand() < 0.5:  # 50% chance to mask the block
                    mask[i:i+block_size, j:j+block_size] = -1

        population.append(mask)
    
    return population

def initialize_population_gaussian(matrix, block_size, population_size, mean=0, stddev=1):
    """
    Initialize a population of masks using Gaussian-distributed values.
    """
    n, m = matrix.shape
    population = []
    
    for population_index in range(population_size):
        # Sample a Gaussian distribution for the mask
        mask = np.random.normal(loc=mean, scale=stddev, size=(n, m))
        
        mask = np.where(mask > 0, 1, -1)  # Values greater than 0 become 1, others become -1
        
        population.append(mask)
    
    return population

def initialize_population_checkerboard(matrix, block_size, population_size):
    """
    Initialize a population of masks using a checkerboard pattern.
    Dynamically adjusts block sizes within the function.
    """
    n, m = matrix.shape
    population = []
    
    # Dynamically generate block sizes by doubling up to the matrix dimensions
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2  # Double the block size for the next iteration
    
    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m))  # Start with all ones

            # Apply checkerboard pattern
            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    # Create alternating pattern: Mask the block every other block
                    if (i // block_size + j // block_size) % 2 == 0:
                        mask[i:i+block_size, j:j+block_size] = -1

            population.append(mask)
    
    return population


def initialize_population_strided_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks by creating strided blocks with varying shifts.
    Dynamically adjusts block sizes within the function.
    """
    n, m = matrix.shape
    population = []
    
    # Dynamically generate block sizes by doubling up to the matrix dimensions
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2  # Double the block size for the next iteration
    
    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m))  # Start with all ones

            # Define the shift step
            step_i = population_index % block_size  # Vary shifts with population index
            step_j = (population_index + 1) % block_size  # Vary shifts with population index

            # Apply the shifts in the block-based regions
            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    shifted_i = (i + step_i) % n
                    shifted_j = (j + step_j) % m

                    # Mask the block
                    mask[shifted_i:shifted_i + block_size, shifted_j:shifted_j + block_size] = -1

            population.append(mask)
    
    return population

def generate_improved_random_masks(existing_masks, num_random, mask_shape, matrix=None, temperature=1.0, decay=0.99, block_size=4, sparsity=0.8, walk_steps=10, keep_top_k=5):
   
    def generate_block_random_mask(mask_shape, block_size):
        n_rows, n_cols = mask_shape
        mask = np.zeros(mask_shape)
        for i in range(0, n_rows, block_size):
            for j in range(0, n_cols, block_size):
                block_value = np.random.choice([-1, 1])
                mask[i:i+block_size, j:j+block_size] = block_value
        return mask

    def generate_gaussian_random_mask(mask_shape, mean=0, std_dev=1):
        return np.random.normal(loc=mean, scale=std_dev, size=mask_shape)

    def generate_sparse_random_mask(mask_shape, sparsity):
        mask = np.random.choice([0, 1], size=mask_shape, p=[sparsity, 1-sparsity])
        return mask * 2 - 1  # Convert 0s to -1s and keep 1s as is

    def generate_pattern_aware_random_mask(mask_shape, matrix):
        mask = np.random.choice([-1, 1], size=mask_shape)
        if matrix is not None:
            high_value_areas = matrix > np.mean(matrix)
            mask[high_value_areas] = 1
        return mask

    def annealed_random_mask(mask_shape, temperature, decay):
        mask = np.random.choice([-1, 1], size=mask_shape)
        for i in range(mask_shape[0]):
            for j in range(mask_shape[1]):
                if np.random.random() > temperature:
                    mask[i, j] = np.random.choice([-1, 1])
        temperature *= decay
        return mask
    
    def random_walk(mask, num_steps=10):
        n_rows, n_cols = mask.shape
        new_mask = mask.copy()
        
        for _ in range(num_steps):
            # Pick a random position in the mask to flip
            i, j = np.random.randint(0, n_rows), np.random.randint(0, n_cols)
            new_mask[i, j] = -new_mask[i, j]  # Flip the selected position
            
        return new_mask

    def generate_hybrid_random_mask(mask_shape, matrix, block_size, sparsity, temperature, decay, walk_steps):
        strategy = np.random.choice(['block', 'gaussian', 'sparse', 'annealed', 'pattern', 'walk'])
        
        if strategy == 'block':
            return generate_block_random_mask(mask_shape, block_size)
        elif strategy == 'gaussian':
            return generate_gaussian_random_mask(mask_shape)
        elif strategy == 'sparse':
            return generate_sparse_random_mask(mask_shape, sparsity)
        elif strategy == 'annealed':
            return annealed_random_mask(mask_shape, temperature, decay)
        elif strategy == 'pattern':
            return generate_pattern_aware_random_mask(mask_shape, matrix)
        elif strategy == 'walk':
            # Start with a random initial mask, then perform a random walk
            initial_mask = np.random.choice([-1, 1], size=mask_shape)
            return random_walk(initial_mask, num_steps=walk_steps)

    # Generate the requested number of random masks using the hybrid strategy
    new_masks = []
    for _ in range(num_random):
        random_mask = generate_hybrid_random_mask(mask_shape, matrix, block_size, sparsity, temperature, decay, walk_steps)
        new_masks.append(random_mask)

    # Rank the new masks based on the fobj score
    ranks = [(i, fobj(matrix * individual)[0]) for i, individual in enumerate(new_masks)]
    
    # Sort by the rank (ascending order)
    sorted_ranks = sorted(ranks, key=lambda x: x[1])  # Sort by the second item (rank)
    
    # Keep only the best `keep_top_k` masks
    best_masks = [new_masks[i] for i, _ in sorted_ranks[:num_random]]
    
    # Add the best masks to the existing ones
    existing_masks.extend(best_masks)

    return existing_masks

def hierarchical_block_tabu_heuristic(matrix, initial_block_size=2, scaling_factor=2):
    n = matrix.shape[0]  # Size of the full matrix

    # Initialize a global mask with all ones
    global_mask = np.ones_like(matrix, dtype=int)

    # Start with the smallest block size and progressively increase
    block_size = initial_block_size
    while block_size <= n:
        num_blocks = int(n // block_size)  # Number of blocks along each dimension
        # Refine the mask using the current block size
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Extract the block
                block = matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                
                # Extract the corresponding portion of the global mask
                block_mask_initial = global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                
                # Apply tabu search to optimize the block
                block_mask, _ = tabu_block(block, block_mask_initial)
                
                # Update the corresponding portion of the global mask
                global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_mask

        # Increase the block size for the next iteration
        block_size *= scaling_factor

    return global_mask

def tabu_block(matrix, initial_mask, tot_resets = 1, num_n = 25, num_m = 1, tabu_size=100, max_no_improve=15, max_iterations=100):

    all_rank_histories = []
    rank_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    next_mask = initial_mask
    best_overall_mask = None
    total_resets = 0
    best_rank = best_overall_rank;
    tabu_set = set()


    while total_resets < tot_resets and best_rank > 2: 
        current_mask = next_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(matrix *current_mask)
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0

        rank_history = []
        for iteration in range(max_iterations):
            num_neighbors = num_n
            num_modifications = max(1, int(num_m * (1 - (iteration//max_iterations)*5)))
            neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)            


            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    masked_matrix = matrix * neighbor
                    neighbor_rank, neighbor_singular_value = fobj(masked_matrix)
                    evaluated_neighbors.append((neighbor_rank, neighbor_singular_value, neighbor_bytes))

            evaluated_neighbors.sort(key=lambda x: (x[0], x[1]))

            if evaluated_neighbors:
                best_neighbor_rank, best_neighbor_singular_value, best_neighbor_bytes = evaluated_neighbors[0]
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape)
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value

                rank_history.append((best_rank, current_singular_value))
                
                if current_rank <= best_rank and compareP1betterthanP2(matrix, current_mask, best_mask):

                    best_mask = current_mask.copy()
                    best_rank = current_rank
                    best_singular_value = current_singular_value
                    no_improve = 0

                else:
                    no_improve += 1                

                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()

            if best_rank == 2:
                break

            if no_improve >= max_no_improve:
                break

        if best_rank < best_overall_rank or (best_rank == best_overall_rank and best_singular_value < best_overall_singular_value):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask

        all_rank_histories.append(rank_history)
        total_resets += 1

    return best_overall_mask, best_overall_rank

def print_top_5(population, matrix, label, num_to_optimise=5):
    unique_population = {mask.tobytes(): i for i, mask in enumerate(population)}  # Deduplicate
    unique_ranks = [(idx, fobj(matrix * population[idx].astype(np.float64))[0]) for _, idx in unique_population.items()]

    # Sort by rank
    sorted_ranks = sorted(unique_ranks, key=lambda x: x[1])

    print(f"Top {num_to_optimise} masks for {label}:")

    selected_indices = []
    for idx, rank in sorted_ranks:
        mask_bytes = population[idx].astype(np.float64).tobytes()  # Get the byte representation of the mask
        print(f"Index: {idx}, Rank: {rank}")
        selected_indices.append(idx)

        if len(selected_indices) == num_to_optimise:
            break

    print("\n")  # Add a newline for better readability
    return selected_indices  # Return indices of the top 5 unique masks

#%%


def tabu_search_with_plot(matrix, initial_mask, tot_resets, num_n, num_m, tabu_size=10, max_no_improve=10, max_iterations=100):
    """
    Recherche tabou pour minimiser le rang avec graphiques.
    """
    # Historique pour tous les voisinages
    all_rank_histories = []
    rank_history = []
    mask_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    best_overall_mask = None
    total_resets = 0

    best_rank = best_overall_rank;
    
    tabu_set = set()

    
    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = initial_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(apply_mask(matrix, current_mask))
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
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape)
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value
        
                rank_history.append((best_rank, current_singular_value))

                if current_rank < best_rank or (current_rank == best_rank and current_singular_value < best_singular_value):
                    if current_rank < best_rank :
                        max_no_improve *= 1.05
                        max_no_improve = min(max_no_improve, 250)
                    best_mask = current_mask.copy()
                    mask_history.append(best_mask)
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
        if best_rank < best_overall_rank or (best_rank == best_overall_rank and best_singular_value < best_overall_singular_value):
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

    return best_overall_mask, best_overall_rank, mask_history


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

def simple_block_mask(block, threshold=1e-2):
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
    original_matrix = read_matrix("test(pas unitaire)/correl5_matrice.txt")
    # original_matrix = matrices1_ledm(120)
    sqrt_matrix = np.sqrt(original_matrix)
    in_rank, s = fobj(sqrt_matrix)
    print("Matrice originale :\n", original_matrix)
    print("Initial Rank :\n", in_rank)
    print("Initial Sing Values :\n", s)
    print("Matrice sq rt :\n", sqrt_matrix)

    # Exemple d'application
    block_size = 6  # Taille des sous-matrices
    threshold = 1e-13  # Seuil pour détecter les contributions faibles

    #block_mask = initialize_mask_from_low_rank_blocks(sqrt_matrix, block_size=5)
    block_mask = initialize_mask_from_blocks(original_matrix, block_size)
    print("Masque basé sur blocs :", block_mask)
    
    initial_population = hybrid_initialize_population(sqrt_matrix, 300)
    best  = print_top_5(initial_population, sqrt_matrix, "Best {idx}", 5)
    
    for idx in best:
        print(initial_population[idx])

        # Appliquer la recherche tabou avec graphique
        start = time.time()
        #best_mask, best_rank = tabu_search_with_plot(sqrt_matrix, initial_mask,mod_neighbors = (0.5, 0.01), mod_mod = (0.025, 0.01), tabu_size=1000, max_no_improve=2000, max_iterations=1000000)
        best_mask, best_rank, tabu_masks = tabu_search_with_plot(
            sqrt_matrix, initial_population[idx],
            tot_resets = 1,
            num_n = 100,
            num_m = 1,
            tabu_size=1000, 
            max_no_improve=50, 
            max_iterations=10000)
        stop = time.time()
        print((stop-start)//60, (stop-start)%60)
        # Résultats finaux
        print("\nMeilleur masque trouvé :\n", best_mask)
    
        print("Rang minimal obtenu :", best_rank)
        print(len(tabu_masks))
        
        # Appliquer le meilleur masque trouvé
        final_matrix = apply_mask(sqrt_matrix, best_mask)