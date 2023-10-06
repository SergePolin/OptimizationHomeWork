import numpy as np
from typing import List

def simplex_method(C, A, b, accuracy=1e-6):
    m, n = len(A), len(C)
    B = np.arange(n, n + m)  # Initial basis
    C_B = np.array([C[i] if i < n else 0 for i in B - n])  # Coefficients of the basic variables
    try:
        X_B = np.linalg.solve(A[:, B - n], b)  # Basic variable values
    except np.linalg.LinAlgError:
        return "The method is not applicable!"
    iteration = 0

    while True:
        B_inv = np.linalg.inv(A[:, B - n])
        z_minus_c = np.dot(C_B, B_inv).dot(A) - C

        if all(z_minus_c >= -accuracy):  # Optimality condition
            x_star = np.zeros(n)
            for i in range(m):
                x_star[B[i] - n] = X_B[i]
            z_star = np.dot(C_B, X_B)
            return x_star, z_star

        j_enter = np.argmin(z_minus_c)
        d = np.dot(B_inv, A[:, j_enter])

        if all(d <= accuracy):  # Unbounded case
            return "The method is not applicable!"

        ratios = [X_B[i] / d[i] if d[i] > 0 else np.inf for i in range(m)]
        i_leave = np.argmin(ratios)

        B[i_leave] = j_enter
        C_B[i_leave] = C[j_enter]
        X_B[i_leave] = b[i_leave] / d[i_leave]

        iteration += 1
        if iteration > 1000:  # Safety check to avoid infinite loop
            return "The method did not converge"


def get_objective_func_coefs_prompt() -> List[float]:
    C = []

    try:
        input_str = input("\nEnter coefficients of the objective funciton, separated by space (for example \"1 2 3\"):\n")
        C = [float(num) for num in input_str.split()]
    except ValueError:
        print("\nInvalid input. Please enter numbers separated by spaces.")
        print("Example: 1 -1 0 2")
        exit()

    return C


def get_constraints_coefs_prompt(variables_num : int) -> np.ndarray:
    try:
        num_rows = int(input("\nEnter the number of constraints functions: "))
    except ValueError:
        print("\nInvalid input. Please enter only one positive number.")
        print("Example: 1")
        exit()

    A = np.empty((num_rows, len(C)), dtype=float)

    for i in range(num_rows):
        try:
            input_str = input(f"Enter coefficients ({variables_num}) of the constraint funciton №{i+1}:\n")
            coefs = [float(num) for num in input_str.split()]

            if len(coefs) != variables_num:
                print(f"Invalid input. Number of coefficients is not equal to the number of variables ({len(coefs)} != {variables_num})")
                exit()

            for j in range(len(A[i])):
                A[i][j] = coefs[j]

        except ValueError:
            print("\nInvalid input. Please enter numbers separated by spaces.")
            print("Example: 1 -1 0 2")
            exit()

    return A


def get_b_vector(constraints_num : int) -> List[float]:
    b = []
    try:
        input_str = input(f"\nEnter b vector (size = {constraints_num}):\n")
        b = [float(num) for num in input_str.split()]
    except ValueError:
        print("\nInvalid input. Please enter numbers separated by spaces.")
        print("Example: 1 -1 0 2")
        exit()

    if len(b) != constraints_num:
        print(f"Invalid input. Number of constraints ({constraints_num}) != size of b (({int(len(b))}.")
        exit()

    return b


def print_prompted_data(C, A, b, accuracy):
    print("\n----------------------------\n")
    print("Objective function coefs:", C)
    print("Constraint functions coefs:\n", A)
    print("B vector:", b)
    print("Accuracy:", accuracy)
    print("\n----------------------------\n")


if __name__ == "__main__":

    C = get_objective_func_coefs_prompt()

    variables_num = len(C)
    A = get_constraints_coefs_prompt(variables_num)

    constaints_num = len(A)
    b = get_b_vector(constaints_num)

    accuracy = 1e-6

    print_prompted_data(C, A, b, accuracy)

    result = simplex_method(C, A, b, accuracy)

    if isinstance(result, str):
        print(result)
    else:
        x_star, z_star = result
        print("Optimal solution:", x_star)
        print("Optimal value:", z_star)

    print("\n----------------------------\n")
