import numpy as np

def print_vector(v):
    """ Function to print a vector. """
    print(" ".join(map(str, v)))

def matrix_vector_product(A, x):
    # Matrix-vector product without using "@" operator
    result = []
    for row in A:
        result.append(sum(a * b for a, b in zip(row, x)))
    return np.array(result)

def solve_linear_system(J, rhs):
    # Solve the linear system without using np.linalg.solve
    n = len(rhs)
    L, U = lu_decomposition(J)
    y = forward_substitution(L, rhs)
    x = backward_substitution(U, y)
    return x

def lu_decomposition(A):
    # LU decomposition without using np.linalg
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            if U[i][i] == 0:
                U[i][i] = 1e-12  # Avoid division by zero
            U[i][k] = A[i][k] - sum_
        
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_ = sum(L[k][j] * U[j][i] for j in range(i))
                if U[i][i] == 0:
                    L[k][i] = 1e-12  # Avoid division by zero
                L[k][i] = (A[k][i] - sum_) / U[i][i]
    
    return L, U

def forward_substitution(L, b):
    # Forward substitution to solve lower triangular system
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        if L[i][i] == 0:
            L[i][i] = 1e-12  # Avoid division by zero
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    
    return y

def backward_substitution(U, y):
    # Backward substitution to solve upper triangular system
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0:
            U[i][i] = 1e-12  # Avoid division by zero
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    
    return x

def interior_point(A, b, C, epsilon, alpha):
    m, n = A.shape
    x = np.ones(n)
    s = np.ones(m)
    e = np.ones(n)
    mu = np.ones(n)
    max_iterations = 1000
    tol = 1e-9

    for _ in range(max_iterations):
        # Define the central path equations
        Ax = matrix_vector_product(A, x)
        F = np.concatenate((Ax - s - b, s * x - alpha * e, mu * s))

        # Check for convergence
        if np.linalg.norm(F) < epsilon:
            return x

        # Construct the system of linear equations
        J = np.zeros((2 * m + n, 2 * m + n))
        J[:m, :n] = A
        J[:m, n:n+m] = -np.eye(m)
        J[:m, n+m:] = np.eye(m)
        J[m:m+n, :n] = np.diag(s)
        J[m:m+n, n:n+m] = np.diag(x)
        J[m:m+n, n+m:] = np.zeros((n, m))
        J[m+n:, :n] = np.zeros((m, n))
        J[m+n:, n:n+m] = np.diag(mu)
        J[m+n:, n+m:] = np.diag(s)

        # Construct the right-hand side of the linear system
        rhs = np.concatenate((b - Ax - s, alpha * e - s * x, np.zeros(n) - mu * s))

        # Solve the linear system using LU decomposition and forward/backward substitution
        dx_affine = solve_linear_system(J, rhs)

        # Split the solution into components
        dx = dx_affine[:n]
        ds = dx_affine[n:n+m]
        dmu = dx_affine[n+m:]

        # Compute step lengths
        if any(ds < 0):
            alpha_p = min(1.0, 0.995 * min([-s[i] / ds[i] for i in range(m) if ds[i] < 0]))
        else:
            alpha_p = 1.0
        if any(dmu < 0):
            alpha_d = min(1.0, 0.995 * min([-mu[i] / dmu[i] for i in range(n) if dmu[i] < 0]))
        else:
            alpha_d = 1.0

        # Compute the centering parameter sigma
        sigma = (sum(s[i] * ds[i] + mu[i] * dmu[i] for i in range(n)) + sum(x[i] * dmu[i] for i in range(n))) / (m + n)

        # Update the variables
        x = x + alpha_p * dx
        s = s + alpha_p * ds
        mu = mu + alpha_d * dmu

    print("The algorithm did not converge.")
    return None

if __name__ == "__main__":
    C = np.array(list(map(float, input("Enter the coefficients of the objective function: ").split())))
    m, n = map(int, input("Enter the size of matrix A (Example: 3 4): ").split())
    print("Enter the coefficients of constraint functions:")
    a = np.array([list(map(float, input().split())) for _ in range(m)])
    b = np.array(list(map(float, input("Enter the right-hand side numbers of constraint functions: ").split())))
    epsilon = float(input("Enter the approximation accuracy: "))
    alpha = 0.5  # You can change the value of alpha as needed
    result = interior_point(a, b, C, epsilon, alpha)
    if result is not None:
        print("The optimal solution is:", result)
