import numpy as np

def calculate_derivative_coefficients(m, k, j1, j2):

    # Creat computational stencil
    n = j1 + j2 + 1

    # Create a matrix A to store the coefficients
    A = np.zeros((n, n))

    # Calculate the central difference coefficients
    for i in range(n):
        for j in range(n):
            if i >= j:
                A[i, j] = (-1) ** (i - j) * np.math.factorial(i) / (np.math.factorial(j) * np.math.factorial(i - j) * (j + 1) ** m)

    # Compute the pseudoinverse of A to solve for b
    A_inv = np.linalg.pinv(A)
    b = np.zeros(n)
    b[j1] = 1  # Derivative term

    # Calculate the coefficients using the pseudoinverse
    b = np.dot(A_inv, b)

    return b[:j1], b[j1], b[j1 + 1:]

# Example usage:
m  = 2  # Desired derivative order
k  = 4  # Cut-off precision order
j1 = 2  # Number of points before point i
j2 = 2  # Number of points after point i

b = calculate_derivative_coefficients(m, k, j1, j2)
print("b =", b)
