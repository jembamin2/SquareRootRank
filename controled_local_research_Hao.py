import numpy as np
import opti_combi_projet_pythoncode_texte as opti
import random
import time

def local_search_sequential(M, max_iterations=10000000):

    m, n = M.shape  # Dimensions of the matrix
    best_solution = []
    # patterns = [np.random.choice([-1, 1], size=(m, n)) for _ in range(5)]
    patterns = [np.ones((m, n)) * -1]
    # current_pattern = pattern_SA.copy()

    for pattern in patterns:

        current_pattern = pattern.copy()
        best_pattern = current_pattern.copy()
        best_fitness = opti.fobj(M, best_pattern)
        print(f"Starting search. Initial rank: {best_fitness[0]}, Initial smallest singular value: {best_fitness[1]}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # Iterate through each element in the matrix
            for i in range(m):
                for j in range(n):
                    # Swap (flip) the element at (i, j)
                    current_pattern[i, j] *= -1
                    
                    # Evaluate the new pattern
                    new_fitness = opti.fobj(M, current_pattern)
                    
                    # If the new pattern is better, keep it
                    if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
                        best_pattern = current_pattern.copy()
                        best_fitness = new_fitness
                        improved = True
                        
                        # Log the improvement
                        print(f"Iteration {iteration + 1}, Improved at ({i}, {j}): Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                        
                        # Exit inner loop to restart from the first element
                        break
                    else:
                        # If no improvement, revert the swap
                        current_pattern[i, j] *= -1
                
                # Restart from the first element if improvement is made
                if improved:
                    break

            # If no improvement in a full pass, terminate the search
            if not improved:
                print(f"No improvement found in iteration {iteration + 1}. Stopping search.")
                break

        best_solution.append((best_pattern, best_fitness))

    return best_solution
        
    # best_pattern = current_pattern.copy()
    # best_fitness = opti.fobj(M, best_pattern)

    # print(f"Starting search. Initial rank: {best_fitness[0]}, Initial smallest singular value: {best_fitness[1]}")
    
    # for iteration in range(max_iterations):
    #     improved = False
        
    #     # Iterate through each element in the matrix
    #     for i in range(m):
    #         for j in range(n):
    #             # Swap (flip) the element at (i, j)
    #             current_pattern[i, j] *= -1
                
    #             # Evaluate the new pattern
    #             new_fitness = opti.fobj(M, current_pattern)
                
    #             # If the new pattern is better, keep it
    #             if opti.compareP1betterthanP2(M, current_pattern, best_pattern):
    #                 best_pattern = current_pattern.copy()
    #                 best_fitness = new_fitness
    #                 improved = True
                    
    #                 # Log the improvement
    #                 print(f"Iteration {iteration + 1}, Improved at ({i}, {j}): Rank {best_fitness[0]}, Smallest Singular Value {best_fitness[1]}")
                    
    #                 # Exit inner loop to restart from the first element
    #                 break
    #             else:
    #                 # If no improvement, revert the swap
    #                 current_pattern[i, j] *= -1
            
    #         # Restart from the first element if improvement is made
    #         if improved:
    #             break

    #     # If no improvement in a full pass, terminate the search
    #     if not improved:
    #         print(f"No improvement found in iteration {iteration + 1}. Stopping search.")
    #         break

    # return best_pattern, best_fitness

# Example Usage
M = opti.matrices1_ledm(120)  # Example matrix


solutions = local_search_sequential(M ,max_iterations=1000)

print("Best Solutions Found:")
for solution in solutions:
    print(solution[1])


# print("Best Pattern Found:")
# print(best_pattern)
# print(f"Rank: {best_fitness[0]}, Smallest Singular Value: {best_fitness[1]}")
