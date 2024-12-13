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

    # Optimiser le masque pour chaque bloc
    masks = [optimize_block_mask(block, threshold) for block, _, _ in blocks]

    # Reconstruire le masque global
    global_mask = reconstruct_global_mask(blocks, masks, matrix.shape)
    return global_mask

# def initialize_population_with_blocks(matrix, block_size, population_size):
#     """
#     Initialize a population of masks using block-based strategies.
#     """
#     n, m = matrix.shape
#     population = []

#     # Helper function to analyze a block
#     def analyze_block(block):
#         rank = np.linalg.matrix_rank(block)
#         return rank <= 2  # Adjust threshold for "low-rank" blocks as needed

#     for _ in range(population_size):
#         mask = np.ones((n, m))  # Start with all ones
#         for i in range(0, n, block_size):
#             for j in range(0, m, block_size):
#                 block = matrix[i:i+block_size, j:j+block_size]
#                 if analyze_block(block):  # If block is low-rank
#                     mask[i:i+block_size, j:j+block_size] = -1
#         population.append(mask)

#     return population

def initialize_population_with_blocks(matrix, block_size, population_size):
    """
    Initialize a population of masks using block-based strategies, where each mask
    shifts the blocks cyclically around the matrix.
    """
    n, m = matrix.shape
    population = []

    # Helper function to analyze a block
    def analyze_block(block):
        rank = np.linalg.matrix_rank(block)
        return rank <= 3  # Adjust threshold for "low-rank" blocks as needed

    for population_index in range(population_size):
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
                if analyze_block(block):
                    mask[shifted_i:shifted_i+block_size, shifted_j:shifted_j+block_size] = -1

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



def perform_crossover(parent1, parent2):
    """
    Choose randomly between uniform and one-point crossover.
    """
    if np.random.rand() > 0.5:
        return crossover_uniform(parent1, parent2)
    else:
        return crossover(parent1, parent2)  # Utilise votre crossover original

def crossover_uniform(parent1, parent2):
    """
    Perform uniform crossover between two parents.
    """
    n_rows, n_cols = parent1.shape
    mask = np.random.rand(n_rows, n_cols) > 0.5  # Masque aléatoire
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    return offspring1, offspring2


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to produce offspring.
    """
    n_rows, n_cols = parent1.shape
    crossover_point = np.random.randint(1, n_rows)  # Split along rows
    offspring1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    offspring2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return offspring1, offspring2

def mutate(mask, mutation_rate, dynamic_rate=None):
    """
    Mutate a mask with various strategies to promote diversification.
    """
    n_rows, n_cols = mask.shape

    # Ajuster dynamiquement le taux de mutation si un seuil est atteint
    if dynamic_rate is not None and dynamic_rate > 0:
        mutation_rate *= dynamic_rate

    num_mutations = int(n_rows * n_cols * mutation_rate)
    mutation_indices = np.random.choice(n_rows * n_cols, size=num_mutations, replace=False)

    for idx in mutation_indices:
        i, j = divmod(idx, n_cols)
        mask[i, j] *= -1

    # Ajout d'une mutation structurelle : permutation des lignes ou colonnes
    if np.random.rand() < 0.2:  # 20% de chance d'ajouter une mutation structurelle
        if np.random.rand() > 0.5:
            # Permutation de lignes
            row1, row2 = np.random.choice(n_rows, size=2, replace=False)
            mask[[row1, row2], :] = mask[[row2, row1], :]
        else:
            # Permutation de colonnes
            col1, col2 = np.random.choice(n_cols, size=2, replace=False)
            mask[:, [col1, col2]] = mask[:, [col2, col1]]

    return mask


def add_random_masks(population, num_random, mask_shape):
    """
    Add random masks to the population to promote diversity.
    """
    for _ in range(num_random):
        random_mask = np.random.choice([-1, 1], size=mask_shape)
        population.append(random_mask)
    return population


def run_genetic_algorithm_with_tabu(matrix, population, max_generations, tabu_resets, tabu_neighbors,
                                    tabu_modifications, tabu_size, max_no_improve, max_iterations,
                                    nb_parents, mutation_rate):
    best_mask = None
    best_rank = float('inf')  # Assurez-vous de suivre la meilleure solution globale

    for generation in range(max_generations):

        # Évaluation de la population
        fitness_scores = [evaluate_fitness(matrix, mask) for mask in population]

        # Sélection des parents avec diversification
        parents = select_parents(population, fitness_scores, num_parents=nb_parents)
        


        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = perform_crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])

        # Mutation avec diversification
        offspring = [mutate(mask, mutation_rate=mutation_rate, dynamic_rate=generation / 10 if generation > 5 else None) for mask in offspring]
        
        # Ajout d'individus aléatoires pour plus de diversité
        offspring = add_random_masks(offspring, num_random=5, mask_shape=matrix.shape)

        # Crée la nouvelle population
        population = parents + offspring[:len(population) - len(parents)]

        # Appliquer la recherche Tabu sur les meilleures solutions
        sorted_indices = np.argsort([score[0] for score in fitness_scores])  # Trier par fitness (rank)
        num_to_optimize = min(1, len(population))  # Optimiser les N meilleures solutions

        for idx in sorted_indices[:num_to_optimize]:
            # Appliquer la recherche Tabu à la meilleure solution
            tabu_mask, tabu_rank = tabu_search_with_plot(
                matrix, population[idx], tabu_resets, tabu_neighbors,
                tabu_modifications, tabu_size, max_no_improve, max_iterations
            )

            # Remplacer l'individu par la solution optimisée par Tabu
            population[idx] = tabu_mask
            if tabu_rank < best_rank:  # Mettre à jour la meilleure solution si nécessaire
                best_rank = tabu_rank
                best_mask = tabu_mask

            if best_rank == 2:  # Arrêter si le rang souhaité est atteint
                break

        # Affichage de la meilleure solution de la génération actuelle
        print(f"Generation {generation + 1}: Best Rank = {best_rank}, Singular Value = {best_rank}")

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

def generate_neighbors_sparse(current_mask, num_neighbors, num_modifications):
    n_rows, n_cols = current_mask.shape
    neighbors = []

    # Get indices of non-zero entries in the matrix
    non_zero_indices = np.argwhere(current_mask != 0)

    for _ in range(num_neighbors):
        # Create a new neighbor by copying the current mask
        neighbor = current_mask.copy()

        # Randomly select `num_modifications` indices to flip from non-zero indices
        selected_indices = np.random.choice(len(non_zero_indices), num_modifications, replace=False)
        flip_positions = non_zero_indices[selected_indices]

        # Flip the selected indices (invert their values)
        for pos in flip_positions:
            i, j = pos  # Get the row and column index
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
    next_mask = initial_mask
    best_overall_mask = None
    total_resets = 0

    best_rank = best_overall_rank;

    tabu_set = set()


    while total_resets < tot_resets and best_rank > 2:  # Limite sur le nombre de voisinages explorés
        # Initialisation pour un nouveau voisinage
        current_mask = next_mask.copy()
        best_mask = current_mask.copy()
        current_rank, current_singular_value = fobj(apply_mask(matrix, current_mask))
        best_rank = current_rank
        best_singular_value = current_singular_value
        no_improve = 0

        rank_history = []
        print(f"=== Exploration du voisinage {total_resets + 1} ===")
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
                    masked_matrix = apply_mask(matrix, neighbor)
                    neighbor_rank, neighbor_singular_value = fobj(masked_matrix)
                    evaluated_neighbors.append((neighbor_rank, neighbor_singular_value, neighbor_bytes))

            # Sort the neighbors by rank, then by singular value
            evaluated_neighbors.sort(key=lambda x: (x[0], x[1]))

            # Prendre le meilleur voisin (ou plusieurs)
            if evaluated_neighbors:
                best_neighbor_rank, best_neighbor_singular_value, best_neighbor_bytes = evaluated_neighbors[0]
                _, _, second_best_neighbor_bytes = evaluated_neighbors[1]
                current_mask = np.frombuffer(best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape)
                current_rank = best_neighbor_rank
                current_singular_value = best_neighbor_singular_value

                rank_history.append((best_rank, current_singular_value))

                #if current_rank < best_rank:
                if current_rank <= best_rank and compareP1betterthanP2(matrix, current_mask, best_mask):

                    best_mask = current_mask.copy()
                    best_rank = current_rank
                    best_singular_value = current_singular_value
                    print(f"new improve: {iteration}, rank = {current_rank}")
                    no_improve = 0

                else:
                    no_improve += 1

                # Add to tabu set
                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()
            
            next_mask = np.frombuffer(second_best_neighbor_bytes, dtype=current_mask.dtype).reshape(current_mask.shape)


            if best_rank == 2:
                break

            #Si aucune amélioration après plusieurs itérations
            if no_improve >= max_no_improve:
                print(f"num iterations: {iteration}, rank = {best_rank}")
                iterations = max_iterations
                break
            


            #print(f"Iteration {iteration + 1}: Current Rank = {current_rank}, Best Rank = {best_rank}")

        # Mettre à jour le meilleur résultat global
        if best_rank < best_overall_rank or (best_rank == best_overall_rank and best_singular_value < best_overall_singular_value):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask
            save_matrix(best_overall_mask, best_overall_rank, best_overall_singular_value)


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

def tabu_block(matrix, initial_mask, tot_resets = 1, num_n = 15, num_m = 1, tabu_size=10, max_no_improve=10, max_iterations=100):
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
        current_rank, current_singular_value = fobj(apply_mask(matrix, current_mask))
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
                    no_improve = 0

                else:
                    no_improve += 1

                # Add to tabu set
                tabu_set.add(best_neighbor_bytes)
                if len(tabu_set) > tabu_size:
                    tabu_set.pop()

            if best_rank == 2:
                break

            if no_improve == int(max_no_improve * 0.75):
                next_mask = current_mask.copy()
            # Si aucune amélioration après plusieurs itérations
            if no_improve >= max_no_improve:
                break


            #print(f"Iteration {iteration + 1}: Current Rank = {current_rank}, Best Rank = {best_rank}")

        # Mettre à jour le meilleur résultat global
        if best_rank < best_overall_rank or (best_rank == best_overall_rank and best_singular_value < best_overall_singular_value):
            best_overall_rank = best_rank
            best_overall_singular_value = best_singular_value
            best_overall_mask = best_mask
            save_matrix(best_overall_mask, best_overall_rank, best_overall_singular_value)


        # Sauvegarder l'historique du voisinage
        all_rank_histories.append(rank_history)

        # Réinitialiser pour un nouveau voisinage
        total_resets += 1

    return best_overall_mask, best_overall_rank

def dynamic_neigh_modulation(iteration, max_iterations, initial_neighbors, initial_modifications):
    """
    Dynamically adjusts the number of neighbors and modifications based on the iteration number.
    """
    progress = iteration / max_iterations
    num_neighbors = int(initial_neighbors * (1 - progress))  # Reduce neighbors as progress increases
    num_modifications = max(1, int(initial_modifications * (1 - progress)))  # Reduce modifications, ensure at least 1
    return num_neighbors, num_modifications


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

def save_matrix(mask, rank, best_singular):
    with open('output.txt', 'w') as fout:
        
        np.savetxt(fout, mask, fmt='%.0f', delimiter=' ')
        
        fout.write(f"{rank}\n{best_singular}")


#%%

# Example Application
if __name__ == "__main__":
    #•original_matrix = matrices1_ledm(120)
    original_matrix = read_matrix("correl5_matrice.txt")
    print("Matrix 1 (50x50, Rank 5):", original_matrix)
    sqrt_matrix = np.sqrt(original_matrix)
    in_rank, s = fobj(sqrt_matrix)

    print("Initial Matrix:\n", original_matrix)
    print("Initial Rank:", in_rank)
    print("Singular Values:", s)

    # Genetic Algorithm Parameters
    population_size = 25
    num_generations = 5
    mutation_rate = 0.15
    num_parents = 15

    # Parameters for Tabu Search
    tabu_resets = 1
    tabu_neighbors = 200
    tabu_modifications = 1
    tabu_size = 100000
    max_no_improve = 100
    max_iterations = 10000

    # Initialiaze with Heuristique Blocks
    hier_mask = hierarchical_block_tabu_heuristic(sqrt_matrix)
    optimized_matrix = apply_mask(sqrt_matrix, hier_mask)
    optimized_rank, _ = fobj(optimized_matrix)
    print("\nOptimized Matrix:\n", optimized_matrix)
    print("Optimized Rank:", optimized_rank)

    # Initialize Population
    initial_population = initialize_population_with_blocks(original_matrix, block_size=2, population_size=population_size)

    # Run Hybrid Algorithm
    start = time.time()
    best_mask, best_rank = run_genetic_algorithm_with_tabu(sqrt_matrix, initial_population, num_generations, tabu_resets, tabu_neighbors,
                                        tabu_modifications, tabu_size, max_no_improve, max_iterations,
                                        num_parents, mutation_rate)
    stop = time.time()

    print((stop-start)//60, (stop-start)%60)

    # Results
    print("\nBest Mask Found:\n", best_mask)
    print("Minimum Rank Achieved:", best_rank)

    # Apply the best mask to the matrix and display results
    optimized_matrix = apply_mask(sqrt_matrix, best_mask)
    optimized_rank, _ = fobj(optimized_matrix)
    print("\nOptimized Matrix:\n", optimized_matrix)
    print("Optimized Rank:", optimized_rank)

