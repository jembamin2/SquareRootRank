import numpy as np
import Tabu_w_Hybride_Initial as Tabu

M = Tabu.read_matrix("file.txt")
M = np.sqrt(M)
if M.shape[0] != M.shape[1]:
    max_dim = max(M.shape)
    new_M = np.zeros((max_dim, max_dim))
    new_M[:M.shape[0], :M.shape[1]] = M
    M = new_M
print(Tabu.fobj(M))