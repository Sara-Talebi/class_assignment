import numpy as np

##---------------------------------------------------------------------------------------##
def generate_random_matrix_np(rows, cols):
    """Generates a NumPy array of size rows x cols with random floating-point numbers."""
    matrix = np.random.rand(rows, cols)
    return matrix
##---------------------------------------------------------------------------------------##
def naive_matrix_multiplication(A_rows, A_cols, B_cols):
    m = A_rows          # Number of rows in A
    n = A_cols          # Number of columns in A (and rows in B)
    p = B_cols          # Number of columns in B
    
    # Initialize result matrix C of size m x p with zeros
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # Perform matrix multiplication
    for i in range(m):             # Outer loop over rows of A
        for j in range(p):         # Middle loop over columns of B
            for k in range(n):     # Inner loop to compute dot product
                C[i][j] += A[i][k] * B[k][j]
    complexity = m * p * n
    return C, complexity
##---------------------------------------------------------------------------------------##
def test_function():
    # Generate random matrices with compatible dimensions
    A_rows, A_cols = 2, 3
    B_rows, B_cols = 3, 2  # B_rows must equal A_cols
    
    A = generate_random_matrix_np(A_rows, A_cols)
    B = generate_random_matrix_np(B_rows, B_cols)
    
    # Perform matrix multiplication using NumPy
    C = np.dot(A, B)
    
    # Display the matrices
    print("Matrix A:")
    print(A)
    
    print("\nMatrix B:")
    print(B)
    
    print("\nMatrix C (A x B):")
    print(C)

    C, complexity = naive_matrix_multiplication(A_rows, A_cols, B_cols)
    print("Complexity: ", complexity)
##---------------------------------------------------------------------------------------##
test_function()
##---------------------------------------------------------------------------------------##
