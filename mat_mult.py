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
