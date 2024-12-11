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

def initialize_population_with_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks using block-based strategies.
    """
    n, m = matrix.shape
    population = []

    # Helper function to analyze a block
    def analyze_block(block):
        rank = np.linalg.matrix_rank(block)
        return rank <= 2  # Adjust threshold for "low-rank" blocks as needed

    for _ in range(population_size):
        mask = np.ones((n, m))  # Start with all ones
        for i in range(0, n, block_size):
            for j in range(0, m, block_size):
                block = matrix[i:i+block_size, j:j+block_size]
                if analyze_block(block):  # If block is low-rank
                    mask[i:i+block_size, j:j+block_size] = -1
        population.append(mask)

    return population


def evaluate_fitness(matrix, mask):
    """
    Evaluate the fitness of a mask.
    """
    masked_matrix = apply_mask(matrix, mask)
    rank, smallest_singular_value = fobj(masked_matrix)
    # Fitness is rank first, then singular value
    return rank, smallest_singular_value

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

def genetic_algorithm(matrix, initial_population, num_generations, mutation_rate, num_parents, elitism=True):
    """
    Genetic algorithm to optimize the mask.
    """
    population = initial_population
    best_mask = None
    best_rank = float('inf')
    best_singular_value = float('inf')
    rank_history = []

    for generation in range(num_generations):
        # Evaluate fitness for all masks
        fitness_scores = [evaluate_fitness(matrix, mask) for mask in population]
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        best_index = sorted_indices[0]

        # Track the best solution
        best_rank, best_singular_value = fitness_scores[best_index]
        best_mask = population[best_index].copy()
        rank_history.append(best_rank)
        print(f"Generation {generation + 1}: Best Rank = {best_rank}, Smallest Singular Value = {best_singular_value}")

        if best_rank == 2:  # Stop early if desired rank is achieved
            break

        # Select parents
        parents = select_parents(population, fitness_scores, num_parents)

        # Generate new population with crossover and mutation
        new_population = []
        for i in range(0, len(parents) - 1, 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            new_population.extend([offspring1, offspring2])

        # Mutate the new population
        new_population = [mutate(mask, mutation_rate) for mask in new_population]

        # Add elitism (keep the best mask from the current generation)
        if elitism:
            new_population[0] = best_mask

        # Replace the old population
        population = new_population

    # Plot rank history
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rank_history)), rank_history, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Rank')
    plt.title('Rank Evolution During Genetic Algorithm')
    plt.grid(True)
    plt.show()

    return best_mask, best_rank

def hybrid_genetic_tabu_search(
    matrix, initial_population, num_generations, mutation_rate, num_parents,
    block_size, tabu_resets, tabu_neighbors, tabu_modifications,
    tabu_size=10, max_no_improve=10, max_iterations=50, elitism=True
):
    """
    Hybrid Genetic Algorithm with Tabu Search.
    """
    population = initial_population
    best_mask = None
    best_rank = float('inf')
    best_singular_value = float('inf')
    rank_history = []

    for generation in range(num_generations):
        # Evaluate fitness for all masks
        fitness_scores = [evaluate_fitness(matrix, mask) for mask in population]
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        best_index = sorted_indices[0]

        # Track the best solution
        best_rank, best_singular_value = fitness_scores[best_index]
        best_mask = population[best_index].copy()
        rank_history.append(best_rank)
        print(f"Generation {generation + 1}: Best Rank = {best_rank}, Smallest Singular Value = {best_singular_value}")

        if best_rank == 2:  # Stop early if desired rank is achieved
            break

        # Select parents
        parents = select_parents(population, fitness_scores, num_parents)

        # Generate new population with crossover and mutation
        new_population = []
        for i in range(0, len(parents) - 1, 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            new_population.extend([offspring1, offspring2])

        # Mutate the new population
        new_population = [mutate(mask, mutation_rate) for mask in new_population]

        # Add elitism (keep the best mask from the current generation)
        if elitism:
            new_population[0] = best_mask

        # Replace the old population
        population = new_population

        # Apply Tabu Search to the top N solutions
        num_to_optimize = min(3, len(population))  # Optimize top 3 masks
        for idx in sorted_indices[:num_to_optimize]:
            tabu_mask, tabu_rank = tabu_search_with_plot(
                matrix, population[idx], tabu_resets, tabu_neighbors,
                tabu_modifications, tabu_size, max_no_improve, max_iterations
            )

            # Replace the individual with the tabu-optimized solution
            population[idx] = tabu_mask
            if tabu_rank < best_rank:  # Update global best if necessary
                best_rank = tabu_rank
                best_mask = tabu_mask
            if best_rank == 2:  # Stop early if desired rank is achieved
                break

    # Plot rank history
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rank_history)), rank_history, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Best Rank')
    plt.title('Rank Evolution During Hybrid Genetic Algorithm + Tabu Search')
    plt.grid(True)
    plt.show()

    return best_mask, best_rank

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



def tabu_search_with_plot(matrix, initial_mask, tot_resets, num_n, num_m, tabu_size=10, max_no_improve=10, max_iterations=100):
    """
    Recherche tabou pour minimiser le rang avec graphiques.
    """
    # Historique pour tous les voisinages
    all_rank_histories = []
    rank_history = []
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

                #if current_rank < best_rank:
                if current_rank <= best_rank and compareP1betterthanP2(matrix, current_mask, best_mask):

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

    return best_overall_mask, best_overall_rank

def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    """
    Dynamically adjusts the number of neighbors and modifications based on the iteration number.
    """
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))  # Reduce neighbors as progress increases
    num_modifications = max(1, int(initial_modifications * (1 - progress)))  # Reduce modifications, ensure at least 1
    return num_neighbors, num_modifications



#%%

# Example Application
if __name__ == "__main__":
    original_matrix = matrices1_ledm(30)
    original_matrix = read_matrix("correl5_matrice.txt")
    sqrt_matrix = np.sqrt(original_matrix)
    in_rank, s = fobj(sqrt_matrix)

    print("Initial Matrix:\n", original_matrix)
    print("Initial Rank:", in_rank)
    print("Singular Values:", s)

    # Genetic Algorithm Parameters
    population_size = 20
    num_generations = 12
    mutation_rate = 0.015
    num_parents = 25
    
    # Parameters for Tabu Search
    tabu_resets = 5
    tabu_neighbors = 70
    tabu_modifications = 1
    tabu_size = 100000
    max_no_improve = 2000
    max_iterations = 1000
    
    
    # Initialize Population
    initial_population = initialize_population_with_blocks(original_matrix, block_size=2, population_size=population_size)
    
    # Run Hybrid Algorithm
    best_mask, best_rank = hybrid_genetic_tabu_search(
        sqrt_matrix, initial_population, num_generations, mutation_rate, num_parents,
        block_size=2, tabu_resets=tabu_resets, tabu_neighbors=tabu_neighbors,
        tabu_modifications=tabu_modifications, tabu_size=tabu_size,
        max_no_improve=max_no_improve, max_iterations=max_iterations
    )
    
    # Results
    print("\nBest Mask Found:\n", best_mask)
    print("Minimum Rank Achieved:", best_rank)
        
    # Apply the best mask to the matrix and display results
    optimized_matrix = apply_mask(sqrt_matrix, best_mask)
    optimized_rank, _ = fobj(optimized_matrix)
    print("\nOptimized Matrix:\n", optimized_matrix)
    print("Optimized Rank:", optimized_rank)

