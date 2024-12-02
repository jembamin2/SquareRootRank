import numpy as np
import opti_combi_projet_pythoncode_texte as opti
import random
import time

def local_search_sequential(M, max_iterations=10000000):
    import time
    import numpy as np

    start = time.time()
    m, n = M.shape  # Dimensions of the matrix
    best_solution = []

    # Start with an initial pattern
    patterns = [np.ones((m, n)) * -1]

    for pattern in patterns:
        current_pattern = pattern.copy()
        best_pattern = current_pattern.copy()
        best_fitness = opti.fobj(M, best_pattern)
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
                        print(f"Iteration {iteration + 1}, Improved at ({i}, {j}): Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                        break  # Restart single-swap
                    else:
                        current_pattern[i, j] *= -1  # Revert the flip

                if improved:
                    break  # Restart single-swap

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

            # Terminate if no improvement in this iteration
            if not improved:
                print(f"No improvement found in iteration {iteration + 1}. Stopping search.")
                break

        # Store the best solution found
        best_solution.append((best_pattern, best_fitness))
        print(f"Time taken: {time.time() - start} seconds")

    return best_solution

# Example Usage
M = opti.matrices1_ledm(120)  # Example matrix


solutions = local_search_sequential(M ,max_iterations=1000000)

print("Best Solutions Found:")
for solution in solutions:
    print(solution[1])


# print("Best Pattern Found:")
# print(best_pattern)
# print(f"Rank: {best_fitness[0]}, Smallest Singular Value: {best_fitness[1]}")
