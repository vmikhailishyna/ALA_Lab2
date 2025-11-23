import numpy as np

def eigenvalues_eigenvectors_matrix(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        A_v = A@v
        lam_v = lam * v
        verity_equality = np.isclose(A_v, lam_v)
        print(f"Verify the equality A⋅v=λ⋅v for eigenvalue - {lam}\neigenvector -{v}\nverity equality is {verity_equality}")
    return eigenvectors, eigenvalues


A = [[4, 2],
     [1, 7]]

eigenvectors, eigenvalues = eigenvalues_eigenvectors_matrix(A)
print("\nEigenvectors", eigenvectors)
print("\nEigenvalues", eigenvalues)
