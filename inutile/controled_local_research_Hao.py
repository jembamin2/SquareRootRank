import numpy as np
import opti_combi_projet_pythoncode_texte_v2 as opti
import random
import time


def read_matrix(input_file):
    with open(input_file, 'r') as fin:
        matrix = []
        r,c=map(int,fin.readline().split())
        for i in range (r):
            tmp = fin.readline().split()
            matrix.append(list(map(float, tmp)))
    return np.array(matrix)


def local_search_sequential(M, max_iterations):
    import time
    import numpy as np
    import random

    start = time.time()
    m, n = M.shape  # Dimensions of the matrix
    best_solution = []

    # Start with an initial pattern
    patterns = [np.ones((m, n))*-1]

    for pattern in patterns:
        current_pattern = pattern.copy()
        best_pattern = current_pattern.copy()
        best_fitness = opti.fobj(M, best_pattern)
        stagnation_counter = 100  # Counter for consecutive stagnation iterations

        print(f"Starting search. Initial rank: {best_fitness[0]}, Initial smallest singular value: {best_fitness[1]}")

        for iteration in range(max_iterations):
            improved = False

            # Single-element flipping
            for i in range(m):
                for j in range(n):
                    current_pattern[i, j] *= -1  # Flip element
                    new_fitness = opti.fobj(M, current_pattern)

                    if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                        best_pattern = current_pattern.copy()
                        best_fitness = new_fitness
                        improved = True
                        stagnation_counter = 0  # Reset stagnation counter
                        print(f"Iteration {iteration + 1}, Improved at ({i}, {j}): Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                        break  # Restart single-swap
                    else:
                        current_pattern[i, j] *= -1  # Revert the flip

                if improved:
                    break  # Restart single-swap

            if improved:
                continue  # Restart single-element flipping

            # Diagonal flipping
            np.fill_diagonal(current_pattern, current_pattern.diagonal() * -1)
            new_fitness = opti.fobj(M, current_pattern)

            if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                best_pattern = current_pattern.copy()
                best_fitness = new_fitness
                improved = True
                stagnation_counter = 0  # Reset stagnation counter
                print(f"Iteration {iteration + 1}, Improved with Diagonal flipping: Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
            else:
                np.fill_diagonal(current_pattern, current_pattern.diagonal() * -1)  # Revert

            if improved:
                continue  # Restart single-element flipping

            # Row-wise flipping
            for i in range(m):
                current_pattern[i, :] *= -1  # Flip row
                new_fitness = opti.fobj(M, current_pattern)

                if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                    best_pattern = current_pattern.copy()
                    best_fitness = new_fitness
                    improved = True
                    print(f"Iteration {iteration + 1}, Improved at Row {i}: Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                    break  # Restart to single-swap
                else:
                    current_pattern[i, :] *= -1  # Revert the flip

            if improved:
                continue  # Restart single-element flipping

            # Column-wise flipping
            for j in range(n):
                current_pattern[:, j] *= -1  # Flip column
                new_fitness = opti.fobj(M, current_pattern)

                if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                    best_pattern = current_pattern.copy()
                    best_fitness = new_fitness
                    improved = True
                    print(f"Iteration {iteration + 1}, Improved at Column {j}: Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                    break  # Restart to single-swap
                else:
                    current_pattern[:, j] *= -1  # Revert the flip

            if improved:
                continue  # Restart single-element flipping

            # Block flipping
            block_size = random.randint(1, min(m, n) // 2)  # Random block size
            x_start = random.randint(0, m - block_size)
            y_start = random.randint(0, n - block_size)
            current_pattern[x_start:x_start + block_size, y_start:y_start + block_size] *= -1
            new_fitness = opti.fobj(M, current_pattern)

            if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                best_pattern = current_pattern.copy()
                best_fitness = new_fitness
                improved = True
                print(f"Iteration {iteration + 1}, Improved with Block flipping: Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
            else:
                current_pattern[x_start:x_start + block_size, y_start:y_start + block_size] *= -1  # Revert

            if improved:
                continue  # Restart single-element flipping

            if not improved:
                
                break
                # Store the best solution found so far
                best_solution.append((best_pattern.copy(), best_fitness))
                print(f"Local minimum detected at iteration {iteration + 1}. Best solution stored. Rank: {best_fitness[0]}, Smallest Singular Value: {best_fitness[1]}")

                # Generate a new starting point (perturbed version of current pattern)
                new_pattern = np.random.choice([1, -1], size=(m, n))
                current_pattern = new_pattern.copy()
                best_pattern = current_pattern.copy()
                best_fitness = opti.fobj(M, best_pattern)

                print(f"Search reinitialized with new pattern.")
        # Store the best solution found
        best_solution.append((best_pattern, best_fitness))
        print(f"Time taken: {time.time() - start} seconds")

    return best_solution


# M = read_matrix("test(pas unitaire)/correl5_matrice.txt")
# M = read_matrix("test(pas unitaire)/slack7gon_matrice.txt")
# M = read_matrix("test(pas unitaire)/synthetic_matrice.txt")
M = opti.matrices2_slackngon(16)  # Example matrix


solutions = local_search_sequential(M ,max_iterations=1000000)

print("Best Solutions Found:")
for solution in solutions:
    print(M.shape[0],solution[1])


# print("Best Pattern Found:")
# print(best_pattern)
# print(f"Rank: {best_fitness[0]}, Smallest Singular Value: {best_fitness[1]}")
