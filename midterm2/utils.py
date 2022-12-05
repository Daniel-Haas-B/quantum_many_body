import numpy as np

def jacobi(A, eps=1e-15, maxiter=10000):
    """
    Jacobi method for diagonalising a symmetric matrix A
    """
    n = A.shape[0]
    V = np.eye(n)
    for i in range(maxiter):
        # find the largest off-diagonal element
        maxval = 0
        for j in range(n):
            for k in range(j+1, n):
                if abs(A[j,k]) > maxval:
                    maxval = abs(A[j,k])
                    p = j # row index
                    q = k # column index
        if maxval < eps:
            break
        # rotate the matrix
        theta = 0.5 * np.arctan2(2*A[p,q], A[p,p]-A[q,q]) # rotation angle
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.eye(n)
        R[p,p] = c
        R[p,q] = -s
        R[q,p] = s
        R[q,q] = c
        A = R.T @ A @ R # matrix rotation
        V = V @ R 
    return A, V # diagonalised matrix and eigenvectors
