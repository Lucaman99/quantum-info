import numpy as np
import math

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
                
                final_vector += np.kron(vector2, vector1)
            
    final_vector = final_vector / np.sqrt(sum(final_vector))
    
    return final_vector

print(Generate_Prime_State(4))
