import numpy as np
import scipy.linalg
import math
import random
from matplotlib import pyplot as plt

# Generating prime states numerically, for a given dimension

def Generate_Prime_State(dimension):
    
    final_vector = np.array([0 for i in range(0, dimension**2)])
    
    for i in range(1, dimension+1):
        for j in range(1, dimension+1):
            
            if (math.gcd(i, j) == 1):
                
                vector1 = np.array([0 for k in range(0, dimension)])
                vector2 = np.array([0 for k in range(0, dimension)])
                
                vector1[i-1] = 1
                vector2[j-1] = 1
                
                final_vector += np.kron(vector1, vector2)
            
    final_vector = final_vector / np.sqrt(sum(final_vector))
    
    return final_vector

print(Generate_Prime_State(3))

# Calculates the partial trace of the matrix

def density_matrix(vector):

    return np.outer(vector, np.conj(vector))

def partial_trace(matrix):
    
    number = int(np.sqrt(len(matrix)))
    new_matrix = []
    
    for i in range(0, number):
        new_row = []
        for j in range(0, number):
            
            sub = matrix[i*number:(i+1)*number, j*number:(j+1)*number]
            new_row.append(np.trace(sub))
        
        new_matrix.append(new_row)
    
    return new_matrix
            
m = np.array([[1, 1, 2, 1], [1, 1, 1, 1], [4, 1, 1, 1], [1, 1, 1, 1]])
print(partial_trace(m))

# Finds the Von Neumann entropy

def entropy(matrix):
    
    log = scipy.linalg.logm(matrix)
    final_val = -1*np.trace(np.matmul(matrix, log))
    return final_val

dimensions = range(2, 50)
y = []
for i in dimensions:
    
    prime_state = Generate_Prime_State(i)
    dm = density_matrix(prime_state)
    new_dm = partial_trace(dm)
    
    y.append(entropy(new_dm))

plt.plot(dimensions, y)
plt.show()
