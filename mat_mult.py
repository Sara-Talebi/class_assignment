import numpy as np
import time

##---------------------------------------------------------------------------------------##
def strassen_matrix_multiplication(A, B, A_rows, B_cols):
    """Performs matrix multiplication using Strassen's algorithm for square matrices."""
    if A_rows != B_cols or A_rows != len(A[0]) or B_cols != len(B[0]):
        raise ValueError("Strassen's algorithm works only for square matrices.")
    
    n = A_rows
    if n == 1:
        return A * B, 1
    
    # Divide matrices into four submatrices
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Strassen's formulas
    M1, M1_complexity = strassen_matrix_multiplication(A11 + A22, B11 + B22, mid, mid)
    M2, M2_complexity = strassen_matrix_multiplication(A21 + A22, B11, mid, mid)
    M3, M3_complexity = strassen_matrix_multiplication(A11, B12 - B22, mid, mid)
    M4, M4_complexity = strassen_matrix_multiplication(A22, B21 - B11, mid, mid)
    M5, M5_complexity = strassen_matrix_multiplication(A11 + A12, B22, mid, mid)
    M6, M6_complexity = strassen_matrix_multiplication(A21 - A11, B11 + B12, mid, mid)
    M7, M7_complexity = strassen_matrix_multiplication(A12 - A22, B21 + B22, mid, mid)
    
    # Combine submatrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 + M3 - M2 + M6
    
    # Combine into result matrix
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    # Calculate complexity
    complexity = 7 * (mid ** 2) + M1_complexity + M2_complexity + M3_complexity + M4_complexity + M5_complexity + M6_complexity + M7_complexity
    return C, complexity
##---------------------------------------------------------------------------------------##
def generate_random_matrix_np(rows, cols):
    """Generates a NumPy array of size rows x cols with random floating-point numbers."""
    matrix = np.random.rand(rows, cols)
    return matrix
##---------------------------------------------------------------------------------------##
def naive_matrix_multiplication(A, B, A_rows, A_cols, B_cols):
    """Performs naive matrix multiplication."""
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
    
    complexity = m * n * p
    return C, complexity
##---------------------------------------------------------------------------------------##
def test_function():
    # Generate random matrices with compatible dimensions for naive and NumPy multiplication
    A_rows, A_cols = 2, 3
    B_rows, B_cols = 3, 2  # B_rows must equal A_cols
    
    A = generate_random_matrix_np(A_rows, A_cols)
    B = generate_random_matrix_np(B_rows, B_cols)
    
    # Perform matrix multiplication using NumPy
    start_time = time.time()
    C_np = np.dot(A, B)
    np_time = time.time() - start_time
    np_complexity = A_rows * A_cols * B_cols  # NumPy uses similar complexity for dot products
    
    # Display the matrices and timings
    print("Matrix A:")
    print(A)
    
    print("\nMatrix B:")
    print(B)
    
    print("\nMatrix C (A x B) using NumPy:")
    print(C_np)
    print(f"\nNumPy np.dot() took {np_time:.6f} seconds.")
    print(f"NumPy complexity: {np_complexity}")

    # Perform naive matrix multiplication
    start_time = time.time()
    C_naive, naive_complexity = naive_matrix_multiplication(A, B, A_rows, A_cols, B_cols)
    naive_time = time.time() - start_time
    
    print("\nMatrix C (A x B) using Naive Multiplication:")
    for row in C_naive:
        print(row)
    print(f"\nNaive matrix multiplication took {naive_time:.6f} seconds.")
    print(f"Naive complexity: {naive_complexity}")

    # Perform Strassen's multiplication (requires square matrices)
    A_square = generate_random_matrix_np(4, 4)
    B_square = generate_random_matrix_np(4, 4)
    
    start_time = time.time()
    C_strassen, strassen_complexity = strassen_matrix_multiplication(A_square, B_square, 4, 4)
    strassen_time = time.time() - start_time
    
    print("\nMatrix C (A x B) using Strassen's Algorithm:")
    print(C_strassen)
    print(f"\nStrassen's algorithm took {strassen_time:.6f} seconds.")
    print(f"Strassen complexity: {strassen_complexity}")
##---------------------------------------------------------------------------------------##
test_function()
##---------------------------------------------------------------------------------------##
