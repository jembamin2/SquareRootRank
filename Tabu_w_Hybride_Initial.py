import matplotlib.pyplot as plt
import numpy as np
import time


#%%

from scipy.linalg import circulant
import os
import datetime
import random

def save_matrix(M, P):
    # Calculate singular values
    sing_values = np.linalg.svd(P * M, compute_uv=False)
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps
    ind_nonzero = np.where(sing_values > tol)[0]

    # Current rank and largest singular value
    current_rank = len(ind_nonzero)
    smallest_singular_value = sing_values[ind_nonzero[-1]]

    # Check for existing files
    existing_files = [f for f in os.listdir('.') if f.startswith("output_rank")]
    # print(len(existing_files), current_rank)
    # random_component = random.randint(1, 999999)
    file_name = False

    for file in existing_files:
        try:
            # Extract rank from the filename
            rank_in_file = int(file.split('_')[1].split('.')[0][4:])  # Extract "5" from "output_rank5_..."
            # print(rank_in_file, " vs ", current_rank)
            if rank_in_file > current_rank:
                # print("here")
                file_name = f"output_rank{current_rank}.txt"

                # Save the new file
                with open(file_name, 'w') as fout:
                    np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
                    for i in ind_nonzero:
                        fout.write(f'{sing_values[i]}\n')
                
                break

            # If the file has the same rank, compare singular values
            elif rank_in_file == current_rank:
                # print(rank_in_file, " vs ", current_rank)
                # print("heerreeee")
                with open(file, 'r') as f:
                    lines = f.readlines()
                    # Read the last singular value in the file
                    last_saved_singular_value = float(lines[-1].strip())
                    # print(f"Last saved singular value: {last_saved_singular_value}")
                    # print(f"Smallest singular value: {smallest_singular_value}")
                    if last_saved_singular_value > smallest_singular_value:
                        file_name = f"output_rank{current_rank}.txt"

                        # Save the new file
                        with open(file_name, 'w') as fout:
                            np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
                            for i in ind_nonzero:
                                fout.write(f'{sing_values[i]}\n')
                break

        except (IndexError, ValueError, FileNotFoundError):
            continue

    if existing_files == []:
        # print("hereAMIE")
        file_name = f"output_rank{current_rank}.txt"

        # Save the new file
        with open(file_name, 'w') as fout:
            np.savetxt(fout, P, fmt='%.0f', delimiter=' ')
            for i in ind_nonzero:
                fout.write(f'{sing_values[i]}\n')
    # if file_name:
        # print(f"Matrix saved to {file_name}")
        
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
    n, m = matrix.shape
    blocks = []
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            blocks.append((block, i, j))  # Garder les indices de début
    return blocks

def optimize_block_mask(block, threshold=1e-2):
    U, S, Vh = np.linalg.svd(block, full_matrices=False)
    mask = np.ones(block.shape)
    for i, singular_value in enumerate(S):
        if singular_value < threshold:
            mask[i, :] = -1  # Marquer les lignes faibles
    return mask

def reconstruct_global_mask(blocks, masks, matrix_shape):
    global_mask = np.ones(matrix_shape)
    for (block, i, j), mask in zip(blocks, masks):
        n, m = mask.shape
        global_mask[i:i+n, j:j+m] = mask
    return global_mask

def initialize_mask_from_blocks(matrix, block_size, threshold=1e-2):
    blocks = split_into_blocks(matrix, block_size)
    masks = [optimize_block_mask(block, threshold) for block, _, _ in blocks]
    global_mask = reconstruct_global_mask(blocks, masks, matrix.shape)
    return global_mask

def generate_neighbors(current_mask, num_neighbors, num_modifications):
    n_rows, n_cols = current_mask.shape
    neighbors = []
    total_elements = n_rows * n_cols
    all_indices = np.arange(total_elements)
    
    for _ in range(num_neighbors):
        neighbor = current_mask.copy()
        flip_indices = np.random.choice(all_indices, num_modifications, replace=False)
        for idx in flip_indices:
            i, j = divmod(idx, n_cols) 
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

    noise_masks = [add_noise_to_mask(mask, noise_rate=0.2) for mask in block_based]

    top_block = sorted(block_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_gaussian = sorted(gaussian_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_checker = sorted(checker_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_strided = sorted(strided_based, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]
    top_noise = sorted(noise_masks, key=lambda m: fobj(matrix * m)[0])[:population_size // 5]

    initial_population = top_block + top_gaussian + top_checker + top_strided + top_noise

    unique_population = ensure_unique_population(initial_population, population_size, matrix)

    select_promising_masks(deduplicate_masks(top_block), matrix, "Block-based")
    select_promising_masks(deduplicate_masks(top_gaussian), matrix, "Gaussian-based")
    select_promising_masks(deduplicate_masks(top_checker), matrix, "Checkerboard-based")
    select_promising_masks(deduplicate_masks(top_strided), matrix, "Strided-based")
    select_promising_masks(deduplicate_masks(top_noise), matrix, "Noise-based")
    select_promising_masks(deduplicate_masks(unique_population), matrix, "Final Initial Population")

    
    return unique_population[:-2] + [initialize_mask_from_blocks(matrix,2)] + [hierarchical_block_tabu_heuristic(matrix)]

def deduplicate_masks(masks):
    unique_masks = set()
    deduplicated = []
    for mask in masks:
        mask_tuple = tuple(map(tuple, mask))
        if mask_tuple not in unique_masks:
            unique_masks.add(mask_tuple)
            deduplicated.append(mask)
    return deduplicated

def ensure_unique_population(masks, population_size, matrix=None, block_size=4):
    unique_masks = set()
    deduplicated = []

    def to_tuple(mask):
        return tuple(map(tuple, mask))  

    for mask in masks:
        mask_tuple = to_tuple(mask)
        if mask_tuple not in unique_masks:
            unique_masks.add(mask_tuple)
            deduplicated.append(mask)

    while len(deduplicated) < population_size:
        additional_masks = generate_improved_random_masks(
            [], 
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
    n, m = matrix.shape
    population = []
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2

    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m))
            block_offset = population_index * block_size % n, population_index * block_size % m

            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    shifted_i = (i + block_offset[0]) % n
                    shifted_j = (j + block_offset[1]) % m

                    block = matrix[shifted_i:shifted_i+block_size, shifted_j:shifted_j+block_size]

                    if np.linalg.matrix_rank(block) <= 3:
                        mask[shifted_i:shifted_i+block_size, shifted_j:shifted_j+block_size] = -1

            population.append(mask)
    
    return population


def initialize_population_random_blocks(matrix, block_size, population_size):
    n, m = matrix.shape
    population = []
    
    for population_index in range(population_size):
        mask = np.ones((n, m))
        for i in range(0, n, block_size):
            for j in range(0, m, block_size):
                if np.random.rand() < 0.5:  
                    mask[i:i+block_size, j:j+block_size] = -1

        population.append(mask)
    
    return population

def initialize_population_gaussian(matrix, block_size, population_size, mean=0, stddev=1):
    n, m = matrix.shape
    population = []
    
    for population_index in range(population_size):
        mask = np.random.normal(loc=mean, scale=stddev, size=(n, m))
        mask = np.where(mask > 0, 1, -1) 
        population.append(mask)
    
    return population

def initialize_population_checkerboard(matrix, block_size, population_size):
    n, m = matrix.shape
    population = []
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2 
    
    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m))

            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    if (i // block_size + j // block_size) % 2 == 0:
                        mask[i:i+block_size, j:j+block_size] = -1

            population.append(mask)
    
    return population


def initialize_population_strided_blocks(matrix, block_size, population_size):
    n, m = matrix.shape
    population = []
    
    block_sizes = []
    current_size = block_size
    while current_size <= min(n, m):
        block_sizes.append(current_size)
        current_size *= 2
    
    for block_size in block_sizes:
        for population_index in range(population_size // len(block_sizes)):
            mask = np.ones((n, m)) 

            step_i = population_index % block_size 
            step_j = (population_index + 1) % block_size  

            for i in range(0, n, block_size):
                for j in range(0, m, block_size):
                    shifted_i = (i + step_i) % n
                    shifted_j = (j + step_j) % m

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
        return mask * 2 - 1 

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
            i, j = np.random.randint(0, n_rows), np.random.randint(0, n_cols)
            new_mask[i, j] = -new_mask[i, j] 
            
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
            initial_mask = np.random.choice([-1, 1], size=mask_shape)
            return random_walk(initial_mask, num_steps=walk_steps)

    new_masks = []
    for _ in range(num_random):
        random_mask = generate_hybrid_random_mask(mask_shape, matrix, block_size, sparsity, temperature, decay, walk_steps)
        new_masks.append(random_mask)

    ranks = [(i, fobj(matrix * individual)[0]) for i, individual in enumerate(new_masks)]
    sorted_ranks = sorted(ranks, key=lambda x: x[1]) 
    best_masks = [new_masks[i] for i, _ in sorted_ranks[:num_random]]
    existing_masks.extend(best_masks)

    return existing_masks

def hierarchical_block_tabu_heuristic(matrix, initial_block_size=2, scaling_factor=2):
    n = matrix.shape[0]  

    global_mask = np.ones_like(matrix, dtype=int)
    block_size = initial_block_size
    while block_size <= n:
        num_blocks = int(n // block_size)  
        for i in range(num_blocks):
            for j in range(num_blocks):
                block = matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                block_mask_initial = global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                block_mask, _ = tabu_block(block, block_mask_initial)
                global_mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_mask
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

def select_promising_masks(population, matrix, label, num_to_optimise=5):
    unique_population = {mask.tobytes(): i for i, mask in enumerate(population)}  
    unique_ranks = [(idx, fobj(matrix * population[idx].astype(np.float64))[0]) for _, idx in unique_population.items()]

    sorted_ranks = sorted(unique_ranks, key=lambda x: x[1])

    print(f"Top {num_to_optimise} masks for {label}:")

    selected_indices = []
    for idx, rank in sorted_ranks:
        print(f"Index: {idx}, Rank: {rank}")
        selected_indices.append(idx)

        if len(selected_indices) == num_to_optimise:
            break

    print("\n") 
    return selected_indices  

#%%


def tabu_search_with_plot(matrix, initial_mask, tot_resets, num_n, num_m, tabu_size=10, max_no_improve=10, max_iterations=100):
    all_rank_histories = []
    rank_history = []
    mask_history = []
    best_overall_rank = float('inf')
    best_overall_singular_value = float('inf')
    best_overall_mask = None
    total_resets = 0

    best_rank = best_overall_rank;
    
    tabu_set = set()

    
    while total_resets < tot_resets and best_rank > 2:  
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
            neighbors = generate_neighbors(current_mask, num_neighbors, num_modifications)

            evaluated_neighbors = []
            for neighbor in neighbors:
                neighbor_bytes = neighbor.tobytes()
                if neighbor_bytes not in tabu_set:
                    masked_matrix = apply_mask(matrix, neighbor)
                    neighbor_rank, neighbor_singular_value = fobj(masked_matrix)
                    evaluated_neighbors.append((neighbor_rank, neighbor_singular_value, neighbor_bytes))
            
            evaluated_neighbors.sort(key=lambda x: (x[0], x[1]))
            
            if evaluated_neighbors:
                best_neighbor_rank, best_neighbor_singular_value, best_neighbor_bytes = evaluated_neighbors[0]
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape)
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value
        
                rank_history.append((best_rank, current_singular_value))

                if current_rank < best_rank or (current_rank == best_rank and current_singular_value < best_singular_value):
                    if current_rank < best_rank :
                        max_no_improve *= 1.05
                        max_no_improve = min(max_no_improve, 700)
                    best_mask = current_mask.copy()
                    mask_history.append(best_mask)
                    best_rank = current_rank
                    save_matrix(matrix, best_mask)
                    best_singular_value = current_singular_value
                    print(f"no improve: {no_improve}, \t rank = {current_rank}, \t s = {best_singular_value}, \t max no improve = {int(max_no_improve)}, \t num _m = {num_modifications}")
                    no_improve = 0
                else:
                    no_improve += 1

                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()
            
            if no_improve >= max_no_improve:
                print(f"num iterations: {iteration}, rank = {best_rank}")
                break

        if best_rank < best_overall_rank or (best_rank == best_overall_rank and best_singular_value < best_overall_singular_value):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask

        all_rank_histories.append(rank_history)
        total_resets += 1


    plt.figure(figsize=(12, 8))
    for i, rank_history in enumerate(all_rank_histories):
        iterations = range(len(rank_history)) 
        ranks = [rank for rank, _ in rank_history]
        plt.plot(iterations, ranks, label=f"Voisinage {i + 1}")
    
    plt.axhline(y=best_overall_rank, color='r', linestyle='--', label="Meilleur rang global")
    plt.xlabel("Itérations (réinitialisées pour chaque voisinage)")
    plt.ylabel("Rang")
    plt.title("Évolution du rang pour chaque voisinage (Recherche Tabou)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_overall_mask, best_overall_rank, mask_history


def reconstruct_mask_from_blocks(matrix, blocks, max_rank=3):

    n, m = matrix.shape
    global_mask = np.ones((n, m))

    for block, i, j, h, w in blocks:
        if np.linalg.matrix_rank(block) <= max_rank:
            global_mask[i:i+h, j:j+w] = -1
    
    return global_mask



def initialize_mask_from_blocks_fast(matrix, block_size, mask_method, **kwargs):
    n_rows, n_cols = matrix.shape
    mask = np.ones_like(matrix)
    
    for i in range(0, n_rows, block_size):
        for j in range(0, n_cols, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            
            block_mask = simple_block_mask(block, **kwargs)
            
            mask[i:i+block_size, j:j+block_size] = block_mask
    
    return mask

def simple_block_mask(block, threshold=1e-2):
    mask = np.ones_like(block)
    mask[np.abs(block) < threshold] = -1
    return mask

def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))
    num_modifications = max(1, int(initial_modifications * (1 - progress)))
    return num_neighbors, num_modifications


#%%
if __name__ == "__main__":
    #original_matrix = read_matrix("synthetic_matrice.txt")
    original_matrix = read_matrix("correl5_matrice.txt")
    #original_matrix = matrices1_ledm(50)
    sqrt_matrix = np.sqrt(original_matrix)
    in_rank, s = fobj(sqrt_matrix)
    print("Matrice originale :\n", original_matrix)
    print("Initial Rank :\n", in_rank)
    print("Initial Sing Values :\n", s)
    print("Matrice sq rt :\n", sqrt_matrix)

    initial_population = hybrid_initialize_population(sqrt_matrix, 1000)
    best  = select_promising_masks(initial_population, sqrt_matrix, "Best", 1)
    
    for idx in best:
        print(initial_population[idx])

        start = time.time()
        best_mask, best_rank, tabu_masks = tabu_search_with_plot(
            sqrt_matrix, initial_population[idx],
            tot_resets = 5,
            num_n = 1000,
            num_m = 3,
            tabu_size=1000, 
            max_no_improve=100, 
            max_iterations=10000)
        stop = time.time()
        print((stop-start)//60, (stop-start)%60)
        
        print("\nMeilleur masque trouvé :\n", best_mask)
        print("Rang minimal obtenu :", best_rank)
    
        final_matrix = apply_mask(sqrt_matrix, best_mask)


#%%
# Exemple : fixer les paramètres par défaut

original_matrix = matrices1_ledm(50)
sqrt_matrix = np.sqrt(original_matrix)
in_rank, s = fobj(sqrt_matrix)
    
default_num_n = 50
default_num_m = 1
default_tabu_size = 1000
default_max_no_improve = 50

# Initialiser des dictionnaires pour stocker les résultats
results = {'num_n': [], 'rank': [], 'time': []}

# Boucle pour tester différentes valeurs de num_n
for num_n in [50, 200, 350, 500]:
    initial_population = hybrid_initialize_population(sqrt_matrix, 1000)
    best  = select_promising_masks(initial_population, sqrt_matrix, "Best", 1)
    
    for idx in best:
        print(initial_population[idx])
        start = time.time()
        best_mask, best_rank, tabu_masks = tabu_search_with_plot(
            sqrt_matrix, initial_population[idx],
            tot_resets=3,
            num_n=num_n,
            num_m=default_num_m,
            tabu_size=default_tabu_size,
            max_no_improve=default_max_no_improve,
            max_iterations=10000
        )
        stop = time.time()
    
        # Stocker les résultats
        results['num_n'].append(num_n)
        results['rank'].append(best_rank)
        results['time'].append(stop - start)
        print(f"num_n = {num_n}, Rang = {best_rank}, Temps = {(stop - start):.2f} sec")


plt.figure(figsize=(10, 6))
plt.plot(results['num_n'], results['rank'], marker='o', linestyle='-', color='b')
plt.title("Influence de num_n sur le Rang obtenu")
plt.xlabel("num_n (nombre de voisins)")
plt.ylabel("Rang minimal obtenu")
plt.grid()
plt.show()

default_num_n = 50
default_num_m = 1
default_tabu_size = 1000
default_max_no_improve = 50

# Initialiser des dictionnaires pour stocker les résultats
results = {'num_m': [], 'rank': [], 'time': []}

# Boucle pour tester différentes valeurs de num_n
for num_m in [1, 2, 3, 4, 5]:
    start = time.time()
    
    initial_population = hybrid_initialize_population(sqrt_matrix, 1000)
    best  = select_promising_masks(initial_population, sqrt_matrix, "Best", 1)
    
    for idx in best:
        print(initial_population[idx])
        best_mask, best_rank, tabu_masks = tabu_search_with_plot(
            sqrt_matrix, initial_population[idx],
            tot_resets=3,
            num_n=default_num_n,
            num_m=num_m,
            tabu_size=default_tabu_size,
            max_no_improve=default_max_no_improve,
            max_iterations=10000
        )
        stop = time.time()
        
        # Stocker les résultats
        results['num_m'].append(num_m)
        results['rank'].append(best_rank)
        results['time'].append(stop - start)
        print(f"num_n = {num_n}, Rang = {best_rank}, Temps = {(stop - start):.2f} sec")
    

plt.figure(figsize=(10, 6))
plt.plot(results['num_m'], results['rank'], marker='o', linestyle='-', color='b')
plt.title("Influence de num_m sur le Rang obtenu")
plt.xlabel("num_m (nombre de modifications)")
plt.ylabel("Rang minimal obtenu")
plt.grid()
plt.show()


results_tabu = {'tabu_size': [], 'rank': [], 'time': []}

for tabu_size in [10, 100, 1000, 10000]:
    initial_population = hybrid_initialize_population(sqrt_matrix, 1000)
    best  = select_promising_masks(initial_population, sqrt_matrix, "Best", 1)
    
    for idx in best:
        print(initial_population[idx])
        start = time.time()
        best_mask, best_rank, tabu_masks = tabu_search_with_plot(
            sqrt_matrix, initial_population[idx],
            tot_resets=3,
            num_n=default_num_n,
            num_m=default_num_m,
            tabu_size=tabu_size,
            max_no_improve=default_max_no_improve,
            max_iterations=10000
        )
        stop = time.time()
        
        results_tabu['tabu_size'].append(tabu_size)
        results_tabu['rank'].append(best_rank)
        results_tabu['time'].append(stop - start)

plt.figure(figsize=(10, 6))
plt.plot(results_tabu['tabu_size'], results_tabu['rank'], marker='o', linestyle='-', color='g')
plt.title("Influence de tabu_size sur le Rang obtenu")
plt.xlabel("tabu_size")
plt.ylabel("Rang minimal obtenu")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_tabu['tabu_size'], results_tabu['time'], marker='s', linestyle='--', color='m')
plt.title("Influence de tabu_size sur le Temps d'exécution")
plt.xlabel("tabu_size")
