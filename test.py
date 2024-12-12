import numpy as np

randMat = np.random.rand(50, 50)

np.savetxt('file.txt', randMat)

print("randMat has been saved to file.txt")