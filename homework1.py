import numpy as np

def simplex_method(C, A, b, accuracy=1e-6):
    # Step 0: Initialize basic feasible solution
    m, n = len(A), len(C)
    B = np.arange(n, n + m)  # Initial basis
    C_B = np.array([C[i] for i in B - n])  # Coefficients of the basic variables
    X_B = np.linalg.solve(A[:, B - n], b)  # Basic variable values
    iteration = 0

    while True:
        # Step 1: Compute the inverse of B
        B_inv = np.linalg.inv(A[:, B - n])

        # Step 2: Compute z_j - c_j for all nonbasic variables
        z_minus_c = np.dot(C_B, B_inv).dot(A) - C

        if all(z_minus_c >= -accuracy):  # Optimality condition
            x_star = np.zeros(n)
            for i in range(m):
                x_star[B[i] - n] = X_B[i]
            z_star = np.dot(C_B, X_B)
            return x_star, z_star

        # Step 3: Find entering variable P_j
        j_enter = np.argmin(z_minus_c)

        # Step 4: Compute B^-1 * P_j
        d = np.dot(B_inv, A[:, j_enter])

        if all(d <= accuracy):  # Unbounded case
            return "The method is not applicable!"

        # Step 5: Find leaving variable P_i using ratio test
        ratios = [X_B[i] / d[i] if d[i] > 0 else np.inf for i in range(m)]
        i_leave = np.argmin(ratios)

        # Step 6: Update basis and basic variables
        B[i_leave] = j_enter
        C_B[i_leave] = C[j_enter]
        X_B[i_leave] = b[i_leave] / d[i_leave]

        iteration += 1
        if iteration > 1000:  # Safety check to avoid infinite loop
            return "The method did not converge"

if __name__ == "__main__":
    C = [2, 3, -1]  # Coefficients of the objective function (maximize 2x1 + 3x2 - x3)
    A = np.array([[1, 1, 1], [2, -1, 1]])  # Coefficients of the constraint functions
    b = [4, 2]  # Right-hand side numbers
    accuracy = 1e-6

    result = simplex_method(C, A, b, accuracy)

    if isinstance(result, str):
        print(result)
    else:
        x_star, z_star = result
        print("Optimal solution:", x_star)
        print("Optimal value:", z_star)
