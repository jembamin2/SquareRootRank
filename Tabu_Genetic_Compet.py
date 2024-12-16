import matplotlib.pyplot as plt
import numpy as np
import time
import random


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
  r1, s1 = fobj2(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj2(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

#     # Define a function to print the top 5 individuals with the best rank for a population
# def print_top_5(population, matrix, label):
#     # Evaluate the rank for each individual
#     ranks = [(i, fobj(matrix * individual)[0]) for i, individual in enumerate(population)]
    
#     # Sort by the rank (ascending order)
#     sorted_ranks = sorted(ranks, key=lambda x: x[1])  # Sort by the second item (rank)
    
#     print(f"Top 5 individuals for {label}:")
    
#     # Print the top 5 individuals with the best rank
#     for i in range(min(5, len(sorted_ranks))):  # Ensure we don't exceed the population size
#         idx, rank = sorted_ranks[i]
#         print(f"Rank: {rank}")
    
#     print("\n")  # New line for better readability


def print_top_5(population, matrix, label, seen_masks=set(), num_to_optimise=5):
    unique_population = {mask.tobytes(): i for i, mask in enumerate(population)}  # Deduplicate
    unique_ranks = [(idx, fobj(matrix * population[idx].astype(np.float64))[0]) for _, idx in unique_population.items()]

    # Sort by rank
    sorted_ranks = sorted(unique_ranks, key=lambda x: x[1])

    print(f"Top {num_to_optimise} masks for {label}:")

    selected_indices = []
    redo = []
    for idx, rank in sorted_ranks:
        mask_bytes = population[idx].astype(np.float64).tobytes()  # Get the byte representation of the mask
        if mask_bytes in seen_masks and len(redo) < num_to_optimise:
           print(f"Redo: {idx}, Rank: {rank}")
           redo.append(idx) 
        if mask_bytes not in seen_masks and len(selected_indices) < num_to_optimise:
            # Print the mask's rank and index
            print(f"Index: {idx}, Rank: {rank}")
            selected_indices.append(idx)

        if len(selected_indices) == num_to_optimise:
            break

    print("\n")  # Add a newline for better readability
    return selected_indices + redo  # Return indices of the top 5 unique masks


from datetime import datetime
import os

def save_matrix(M, P):
    # Calculate singular values
    sing_values = np.linalg.svd(P * M, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]

    # Current rank and largest singular value
    current_rank = len(ind_nonzero)
    smallest_singular_value = sing_values[0]

    # Check for existing files
    existing_files = [f for f in os.listdir('.') if f.startswith("output_rank")]

    for file in existing_files:
        try:
            # Extract rank from the filename
            rank_in_file = int(file.split('_')[1][4:])  # Extract "5" from "output_rank5_..."
            if rank_in_file < current_rank:
                print("Already found better")
                return  # Exit without saving

            # If the file has the same rank, compare singular values
            if rank_in_file == current_rank:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    # Read the last singular value in the file
                    last_saved_singular_value = float(lines[-1].strip())
                    if last_saved_singular_value<= smallest_singular_value:
                        print("Not better")
                        return  # Exit without saving
        except (IndexError, ValueError, FileNotFoundError):
            continue  # Skip files that don't match the pattern or have errors

    # Generate a unique filename with date, time, and a random component
    date_today = datetime.now()
    random_component = random.randint(1, 99999)
    file_name = f"output_rank{current_rank}_{date_today}_{random_component}.txt"

    # Save the new file
    with open(file_name, 'w') as fout:
        np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
        for i in ind_nonzero:
            fout.write(f'{sing_values[i]}\n')

    print(f"Matrix saved to {file_name}")



#%%

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

def optimize_block_mask(block, threshold=1e-14):
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

def initialize_mask_from_blocks(matrix, block_size, threshold=1e-14):
    # Diviser la matrice en blocs
    blocks = split_into_blocks(matrix, block_size)

    # Optimiser le masque pour chaque bloc
    masks = [optimize_block_mask(block, threshold) for block, _, _ in blocks]

    # Reconstruire le masque global
    global_mask = reconstruct_global_mask(blocks, masks, matrix.shape)
    return global_mask

#%% Initialization de la population
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


def initialize_population_with_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks using block-based strategies.
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
    """
    Recherche tabou pour minimiser le rang avec graphiques.
    """
    # Historique pour tous les voisinages
    all_rank_histories = []
    rank_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    next_mask = initial_mask
    best_overall_mask = None
    total_resets = 0
    best_rank = best_overall_rank;
    tabu_set = set()


    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = next_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(matrix *current_mask)
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0

        rank_history = []
        for iteration in range(max_iterations):
            num_neighbors, num_modifications = dynamic_neigh_modulation(iteration, max_iterations, num_n, num_m)
            # si générale
            #neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)            # Générer plusieurs voisins
            # si sparse
            neighbors = generate_neighbors_sparse(current_mask, num_neighbors, num_modifications)            # Générer plusieurs voisins


            # Évaluer tous les voisins
            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    masked_matrix = matrix * neighbor
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
                
                if current_rank <= best_rank and compareP1betterthanP2(matrix, current_mask, best_mask):

                    best_mask = current_mask.copy()
                    best_rank = current_rank
                    best_singular_value = current_singular_value
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

    return best_overall_mask, best_overall_rank

#%% Fonctions pour la Génétique


def select_parents(population, fitness_scores, num_parents):
    """
    Select parents based on fitness scores (tournament selection).
    """
    selected_parents = []
    for _ in range(num_parents):
        # Randomly sample a subset and pick the best
        competitors = np.random.choice(len(population), size=3, replace=False)
        best_idx = min(competitors, key=lambda i: fitness_scores[i])
        selected_parents.append(population[best_idx])
    return selected_parents

def crossover(parent1, parent2):
    """
    Perform crossover between two parents to produce offspring.
    """
    n_rows, n_cols = parent1.shape
    crossover_point = np.random.randint(1, n_rows)  # Split along rows
    offspring1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    offspring2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return offspring1, offspring2


def mutate(mask, mutation_rate):
    """
    Mutate a mask by flipping random entries with a given probability.
    """
    n_rows, n_cols = mask.shape
    num_mutations = int(n_rows * n_cols * mutation_rate)
    mutation_indices = np.random.choice(n_rows * n_cols, size=num_mutations, replace=False)
    for idx in mutation_indices:
        i, j = divmod(idx, n_cols)
        mask[i, j] *= -1
    return mask

def diversify_population(matrix, population_size, mask_shape, block_size):
    population = []
    for _ in range(population_size // 2):  # Random masks
        random_mask = np.random.choice([-1, 1], size=mask_shape)
        population.append(random_mask)
    for _ in range(population_size // 2):  # Structured masks
        mask = np.ones(mask_shape)
        for i in range(0, mask_shape[0], block_size):
            for j in range(0, mask_shape[1], block_size):
                if np.random.rand() > 0.7:
                    mask[i:i+block_size, j:j+block_size] *= -1
        population.append(mask)
    return population

def select_diverse_parents(population, fitness_scores, num_parents):
    selected_parents = []
    population_indices = set(range(len(population))) 

    while len(selected_parents) < num_parents:
        current_indices = list(population_indices)
        num_candidates = min(5, len(current_indices))
        candidates = np.random.choice(current_indices, size=num_candidates, replace=False)

        candidate_fitness = [fitness_scores[i][0] for i in candidates]

        best_idx_in_candidates = np.argmin(candidate_fitness)
        best_candidate_idx = candidates[best_idx_in_candidates]

        selected_parents.append(population[best_candidate_idx])

        if best_candidate_idx in population_indices:
            population_indices.remove(best_candidate_idx)

    return selected_parents


def dynamic_mutation(mask, mutation_rate, large_mutation_prob=0.1):
    n_rows, n_cols = mask.shape
    num_mutations = int(n_rows * n_cols * mutation_rate)
    mutation_indices = np.random.choice(n_rows * n_cols, size=num_mutations, replace=False)

    for idx in mutation_indices:
        i, j = divmod(idx, n_cols)
        mask[i, j] *= -1

    # Occasionally perform a large mutation
    if np.random.rand() < large_mutation_prob:
        if np.random.rand() > 0.5:  # Row swap
            row1, row2 = np.random.choice(n_rows, size=2, replace=False)
            mask[[row1, row2], :] = mask[[row2, row1], :]
        else:  # Column swap
            col1, col2 = np.random.choice(n_cols, size=2, replace=False)
            mask[:, [col1, col2]] = mask[:, [col2, col1]]

    return mask


def structured_crossover(parent1, parent2, threshold=1e-14): #FIXME c'est bizarre non ? tu degages une ligne/colonne
    
    def rank_impact(matrix):
        rows, cols = matrix.shape
        original_rank = np.linalg.matrix_rank(matrix)
        
        # Rank impact for rows
        row_impact = []
        for i in range(rows):
            reduced_matrix = np.delete(matrix, i, axis=0)  # Remove row i
            reduced_rank = np.linalg.matrix_rank(reduced_matrix)
            row_impact.append(original_rank - reduced_rank)
        
        # Rank impact for columns
        col_impact = []
        for j in range(cols):
            reduced_matrix = np.delete(matrix, j, axis=1)  # Remove column j
            reduced_rank = np.linalg.matrix_rank(reduced_matrix)
            col_impact.append(original_rank - reduced_rank)
        
        return np.array(row_impact), np.array(col_impact)

    n_rows, n_cols = parent1.shape
    offspring1, offspring2 = parent1.copy(), parent2.copy()

    # Get row and column impacts for both parents
    r1, c1 = rank_impact(parent1)
    r2, c2 = rank_impact(parent2)

    # Identify rows to switch based on impact for parent1 and parent2
    for i in range(n_rows):
        if r1[i] < threshold:  # Select rows with low impact for parent1
            offspring1[i, :] = parent2[i, :]
        if r2[i] < threshold:  # Select rows with low impact for parent2
            offspring2[i, :] = parent1[i, :]

    # Identify columns to switch based on impact for parent1 and parent2
    for j in range(n_cols):
        if c1[j] < threshold:  # Select columns with low impact for parent1
            offspring1[:, j] = parent2[:, j]
        if c2[j] < threshold:  # Select columns with low impact for parent2
            offspring2[:, j] = parent1[:, j]

    return offspring1, offspring2

def add_random_masks(existing_masks, num_random, mask_shape):
    for _ in range(num_random):
        # Create a random mask of 0s and 1s
        random_mask = np.random.choice([-1, 1], size=mask_shape)
        existing_masks.append(random_mask)
    
    return existing_masks

def add_noise_to_mask(mask, noise_rate=0.2):
    noisy_mask = mask.copy()
    total_elements = mask.size
    num_noisy_elements = int(noise_rate * total_elements)

    # Randomly choose indices to flip
    noisy_indices = np.random.choice(total_elements, num_noisy_elements, replace=False)

    # Flatten the mask and flip the chosen elements
    flattened_mask = noisy_mask.flatten()
    flattened_mask[noisy_indices] = -1 * flattened_mask[noisy_indices]  # Flip 0 to 1 and 1 to 0
    
    return flattened_mask.reshape(mask.shape)


def hybrid_genetic_tabu_search(matrix, initial_population, num_generations, mutation_rate, num_parents, block_size, tabu_resets, tabu_neighbors, tabu_modifications, tabu_size=10, max_no_improve=10, max_iterations=50, elitism=False):
    """
    Hybrid Genetic Algorithm with Tabu Search.
    """
    tabu_set = set()
    seen_masks  = set()
    population = initial_population
    pop_size = len(population)
    best_mask = np.ones(matrix.shape)
    best_rank = float('inf')
    best_singular_value = float('inf')
    rank_history = []
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    plt.figure(figsize=(10, 6))
    for generation in range(num_generations):
        
        # Apply Tabu Search to the top N solutions
        count = 0
        population = sorted(population, key=lambda m: fobj(matrix * m)[0])[:pop_size]

        values = [fobj(matrix * mask) for mask in population]
        sorted_indices = np.argsort([score[0] for score in values])  # Trier par fitness (rank)
        
        num_to_optimize = 5
        top_5_indices  = print_top_5(population, matrix, f"Gen {generation + 1}", seen_masks, num_to_optimize)

        random_indices = random.sample([i for i in range(len(population)) if i not in top_5_indices], 5)

        # Combine top 5 and 5 random masks
        all_selected_indices = top_5_indices + random_indices

        gen_rank = matrix.shape[0]
        new_pop = []
        for idx in all_selected_indices:
            mask_bytes = population[idx].astype(np.float64).tobytes()
            if mask_bytes not in seen_masks:
                seen_masks.add(mask_bytes)
            print(f" \n Testing Mask {count + 1} ({idx}) with rank {fobj(matrix * population[idx])[0]}")
            tabu_masks, tabu_rank, rank_progress = tabu_search(
                    tabu_set, matrix, 
                    population[idx].astype(np.float64), 
                    max(1, tabu_resets - (generation*3) - count*2), tabu_neighbors,
                    tabu_modifications*((generation + 1)//3), tabu_size, max_no_improve, max_iterations)
            count += 1

            for mask in tabu_masks:
                mask_rank, _ = fobj(matrix * mask)
             
                if mask_rank < gen_rank:
                    gen_rank = mask_rank
                    best_mask = mask.copy()
                    print(f"Updated Best Mask with rank {gen_rank}")
                    
                    if tabu_rank == 2:
                        return best_mask, tabu_rank
        
            # Extending the new population with the updated tabu masks
            new_pop.extend(tabu_masks)


            # Plot rank progression for this mask
            color = colors[generation % len(colors)]  # Assign color for this generation
            plt.plot(range(len(rank_progress)), rank_progress, label=f'Gen {generation + 1} Mask {count}', color=color)
        
        population.extend(new_pop)
        # Limiter la taille de la population pour préserver la diversité
        population = sorted(population, key=lambda m: fobj(matrix * m)[0])[:population_size]
        
        # Evaluate fitness for all masks
        values = [fobj(matrix * mask) for mask in population]
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        best_index = sorted_indices[0]

        # Track the best solution
        best_rank, best_singular_value = values[best_index]
        best_mask = population[best_index].copy()
        rank_history.append(best_rank)
        save_matrix(matrix, best_mask)
        print(f"Generation {generation + 1}: Best Rank = {gen_rank}, Smallest Singular Value = {best_singular_value} \n \n ")
        print(f"Population diversity: {len(set([mask.tobytes() for mask in population]))} unique masks for {len(population)}")
        
        seen_masks_parents = [np.frombuffer(mask_bytes, dtype=np.float64).reshape(population[0].shape) for mask_bytes in seen_masks]
        population.extend(seen_masks_parents)  # Ajouter les masques de seen_masks à la population

        # Select parents
        values = [fobj(matrix * mask) for mask in population]
        parents = select_diverse_parents(population, values, num_parents)

        # Generate new population with crossover and mutation
        new_population = []
        for i in range(0, len(parents) - 1, 2):
            offspring1, offspring2 = structured_crossover(parents[i], parents[i + 1])
            # offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            new_population.extend([offspring1, offspring2])

        if len(rank_history) >= 2 and rank_history[-1] == rank_history[-2]:
            print("Stagnation détectée, réinitialisation partielle de la population.")
            num_random = int(population_size * 0.1)
            for _ in range(num_random):
                random.shuffle(initial_population)
                population.extend(initial_population[:num_random])

        # Mutate the new population
        new_population = [dynamic_mutation(mask, mutation_rate) for mask in new_population]

        # Add elitism (keep the best mask from the current generation)
        if elitism:
            new_population[0] = best_mask
            
        # Replace the old population        
        population.extend(diversify_population(matrix, pop_size//4, matrix.shape, block_size=2))

        if len(rank_history) >= 2 and rank_history[-1] == rank_history[-2]:
            population.extend(add_random_masks(population, num_random=20, mask_shape=matrix.shape))
        
        new_population = sorted(population, key=lambda m: fobj(matrix * m)[0])[:pop_size//2]

        remaining_population = population[pop_size//2:]
        random.shuffle(remaining_population)
        new_population.extend(remaining_population[:pop_size//2])

        population = ensure_unique_population(new_population, pop_size, matrix)
        

    # Finalize the plot
    plt.title('Rank Progression Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Rank')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.yticks(range(best_rank, matrix.shape[0] + 1)) 
    plt.axhline(y=best_rank, color='gray', linestyle='--', linewidth=1) 
    plt.show()


    return best_mask, best_rank

# def generate_neighbors(current_mask, num_neighbors, num_modifications):
#     """
#     Generates neighbors by flipping a random set of entries in the current_mask.
#     """
#     n_rows, n_cols = current_mask.shape
#     neighbors = []

#     # Randomly pick `num_modifications` positions to flip
#     total_elements = n_rows * n_cols
#     all_indices = np.arange(total_elements)

#     for _ in range(num_neighbors):
#         # Create a new neighbor by copying the current mask
#         neighbor = current_mask.copy()

#         # Randomly select `num_modifications` indices to flip
#         flip_indices = np.random.choice(all_indices, num_modifications, replace=False)

#         # Flip the selected indices (invert their values)
#         for idx in flip_indices:
#             i, j = divmod(idx, n_cols)  # Convert flat index to 2D (i, j)
#             neighbor[i, j] *= -1

#         neighbors.append(neighbor)

#     return neighbors

def generate_neighbors_sparse(current_mask, num_neighbors, num_modifications):

    non_zero_indices = np.nonzero(current_mask) 
    non_zero_elements = len(non_zero_indices[0]) 
    
    if non_zero_elements < num_modifications:
        raise ValueError("Que des 0s")
    
    neighbors = []

    for _ in range(num_neighbors):
        neighbor = current_mask.copy()
        flip_indices = np.random.choice(non_zero_elements, num_modifications, replace=False)

        for idx in flip_indices:
            i, j = non_zero_indices[0][idx], non_zero_indices[1][idx]
            neighbor[i, j] = -neighbor[i, j] 

        neighbors.append(neighbor)

    return neighbors

def tabu_search(tabu_set, matrix, initial_mask, tot_resets, num_n, num_m, tabu_size=10, max_no_improve=10, max_iterations=100):
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
    max_no_improve_d = max_no_improve

    best_rank = best_overall_rank;
    
    rank_progress = []
    current_rank = fobj(matrix *initial_mask)[0]
    rank_progress.append(current_rank)

    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = initial_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj((matrix * current_mask))
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0
        
        print(f"\n === Exploration du voisinage {total_resets + 1} ===")
        for iteration in range(max_iterations):
            num_neighbors, num_modifications = dynamic_neigh_modulation(iteration, max_iterations, num_n, num_m)
            neighbors = generate_neighbors_sparse(current_mask, num_neighbors, num_modifications)            # Générer plusieurs voisins
            # neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)


            # Évaluer tous les voisins
            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    neighbor_rank, neighbor_singular_value = fobj(matrix *neighbor)
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
                best_singular_value = current_singular_value
                best_mask = current_mask.copy()
                best_rank = current_rank
                print(f"no improve: {no_improve}, \t rank = {current_rank}, \t s = {best_singular_value}, \t max no improve = {int(max_no_improve)}, \t num _m = {num_modifications}")
                rank_progress.append(best_rank)
                mask_history.append(best_mask)
                if current_rank - best_rank <= 1:
                    max_no_improve *= 1.05
                    max_no_improve = min(max_no_improve, 250)
                no_improve = 0
            else:
                no_improve += 1


                # Add to tabu set
                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()

            if current_rank == 2:
                iteration = max_iterations
                total_resets = tabu_resets
                mask_history = [current_mask.copy()]
                break

            # Si aucune amélioration après plusieurs itérations
            if no_improve >= max_no_improve:
                print(f"num iterations: {iteration}, rank = {best_rank}")
                mask_history.append(best_mask)
                save_matrix(matrix, best_mask)
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
        max_no_improve = max_no_improve_d


    return mask_history, best_overall_rank, rank_progress



def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    """
    Dynamically adjusts the number of neighbors and modifications based on the iteration number.
    """
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))  # Reduce neighbors as progress increases
    num_modifications = max(1, int(initial_modifications * (1 - progress*5)))  # Reduce modifications, ensure at least 1
    return num_neighbors, num_modifications





#%%

# Example Application
if __name__ == "__main__":
    original_matrix = matrices1_ledm(16)
    #original_matrix = read_matrix("correl5_matrice.txt")
    #original_matrix = read_matrix("synthetic_matrice.txt")
    sqrt_matrix = np.sqrt(original_matrix)
    in_rank, s = fobj(sqrt_matrix)

    print("Initial Matrix:\n", original_matrix)
    print("Initial Rank:", in_rank)
    print("Singular Values:", s)

    # Genetic Algorithm Parameters
    population_size = 300
    num_generations = 10
    mutation_rate = 0.3
    num_parents = 200
    
    # Parameters for Tabu Search
    tabu_resets = 7
    tabu_neighbors = 200
    tabu_modifications = 1
    tabu_size = 100000
    max_no_improve = 50
    max_iterations = 1000 

    initial_population = hybrid_initialize_population(sqrt_matrix, population_size)
    
    # Run Hybrid Algorithm
    start = time.time()
    best_mask, best_rank = hybrid_genetic_tabu_search(
        sqrt_matrix, initial_population, num_generations, mutation_rate, num_parents,
        block_size=2, tabu_resets=tabu_resets, tabu_neighbors=tabu_neighbors,
        tabu_modifications=tabu_modifications, tabu_size=tabu_size,
        max_no_improve=max_no_improve, max_iterations=max_iterations
    )
    stop = time.time()

    # Results
    print("\nBest Mask Found:\n", best_mask)
    print("Minimum Rank Achieved:", best_rank)

    # Apply the best mask to the matrix and display results
    optimized_rank, _ = fobj(sqrt_matrix *best_mask)
    print("\nOptimized Matrix:\n", sqrt_matrix *best_mask)
    print("Optimized Rank:", optimized_rank)
    
    # Check if better
    check_mask = hierarchical_block_tabu_heuristic(sqrt_matrix)
    optimized_rank, _ = fobj(sqrt_matrix *check_mask)
    print("\nOptimized Matrix:\n", sqrt_matrix *check_mask)
    print("Optimized Rank:", optimized_rank)

    print((stop-start)//60, (stop-start)%60)