import numpy as np

def generate_random_matrix_np(rows, cols):
    """Generates a NumPy array of size rows x cols with random floating-point numbers."""
    matrix = np.random.rand(rows, cols)
    return matrix

# Example usage:
rows = 3
cols = 4
A = generate_random_matrix_np(rows, cols)
print("Matrix A:")
print(A)

def naive_matrix_multiplication(A, B):
    m = len(A)          # Number of rows in A
    n = len(A[0])       # Number of columns in A (and rows in B)
    p = len(B[0])       # Number of columns in B
    
    # Initialize result matrix C of size m x p with zeros
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # Perform matrix multiplication
    for i in range(m):         # Outer loop over rows of A
        for j in range(p):     # Middle loop over columns of B
            for k in range(n): # Inner loop to compute dot product
                C[i][j] += A[i][k] * B[k][j]
    
    return C
