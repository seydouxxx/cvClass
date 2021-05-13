import numpy as np

a = np.array([-1, 2, 5])
b = np.array([[-1], [2], [5]])
M = np.zeros((4,3))

def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = None
    ### MY CODE HERE
    out = np.dot(a, b)

    # check if out is scalar
    if (out.shape == (1, 1)):   
        out = out[0][0]
    if (out.shape == (1, )):
        out = out[0]
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    ### MY CODE HERE
    out = dot_product(dot_product(a, b), dot_product(M, a.T))
    ### END YOUR CODE

    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return u, s, v

def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return singular_values

def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    ### MY CODE HERE
    w, v = np.linalg.eig(M)
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    ### MY CODE HERE
    eigenvalues, eigenvectors = eigen_decomp(M)
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors)

    # sort both arrays by order of absolute value of eigenvalues array
    # and return k values with list slicing
    order = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order][:k]
    eigenvectors = eigenvectors[order][:k]
    ### END YOUR CODE
    return eigenvalues, eigenvectors

print("*** 1.A")
print("M = \n", M)
print("The size of M is: ", M.shape)
print()
print("a = \n", a)
print("The size of a is: ", a.shape)
print()
print("b = \n", b)
print("The size of b is: ", b.shape)
print()

print("*** 1.B")
aDotb = dot_product(a, b)
print(aDotb)
print(f'The size is : {aDotb.shape}')
print()

print("*** 1.C")
# ans = complicated_matrix_function(M, a, b)
# print(ans)
# print(f'The size is : {ans.shape}')
# print()

M_2 = np.array(range(4)).reshape((2, 2))
print(M_2)
a_2 = np.array([[1, 1]])
b_2 = np.array([[10, 10]]).T
print(M_2.shape)
print(a_2.shape)
print(b_2.shape)
print()
ans = complicated_matrix_function(M_2, a_2, b_2)
print(ans)
print(f'The size is : {ans.shape}')
print()

print("*** 1.D")
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
val, vec = get_eigen_values_and_vectors(M[:, :3], 1)
print(f'First eigenvalue = {val[0]}')
print(f'First eigenvector = {vec[0]}')
print()
assert len(vec) == 1

val, vec = get_eigen_values_and_vectors(M[:, :3], 2)
print(f'Eigenvalues = {val}')
print(f'Eigenvectors = {vec}')
assert len(vec) == 2
print()