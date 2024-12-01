import numpy as np

x = np.array([[1, 2, 3,4], [4, 5, 6,4], [7, 8, 9,10]])

y = np.linalg.norm(x, ord='fro')

print(y)  # Output: 16.881943016134134

print(np.random.choice([1,-1], size=(5,5)) )