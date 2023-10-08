import numpy as np

# A function to print a vector
def print_vector(v):
    print(" ".join(map(str, v)))

# A function to print a matrix
def print_matrix(m):
    for row in m:
        print_vector(row)
    print()

# A function to find the pivot column in the simplex tableau
def find_pivot_column(c, epsilon):
    pivot_col = -1  # initialize with -1 to indicate no pivot column
    min_c = 0       # initialize with 0 to find the most negative coefficient
    for j in range(len(c)):
        if c[j] < min_c - epsilon:
            min_c = c[j]
            pivot_col = j
    return pivot_col

# A function to find the pivot row in the simplex tableau
def find_pivot_row(a, b, pivot_col, epsilon):
    pivot_row = -1          # initialize with -1 to indicate no pivot row
    min_ratio = float('inf') # initialize with infinity to find the minimum ratio
    for i in range(len(a)):
        if a[i][pivot_col] > epsilon:
            ratio = b[i] / a[i][pivot_col]
            if ratio < min_ratio - epsilon:
                min_ratio = ratio
                pivot_row = i
    return pivot_row

# A function to perform the pivot operation on the simplex tableau
def pivot(a, b, c, pivot_row, pivot_col):
    m, n = a.shape

    # Divide the pivot row by the pivot element
    pivot_element = a[pivot_row, pivot_col]
    a[pivot_row, :] /= pivot_element
    b[pivot_row] /= pivot_element

    # Subtract multiples of the pivot row from other rows
    for i in range(m):
        if i != pivot_row:
            factor = a[i, pivot_col]
            a[i, :] -= factor * a[pivot_row, :]
            b[i] -= factor * b[pivot_row]

    # Update the objective function coefficients
    factor = c[pivot_col]
    c -= factor * a[pivot_row, :]

# A function to implement the simplex method for linear programming problems
def simplex(a, b, c, epsilon):
    m, n = a.shape

    # Create an initial feasible solution by adding slack variables
    tableau = np.zeros((m, n + m))
    for i in range(m):
        tableau[i, :n] = a[i, :]
        tableau[i, n + i] = 1

    z = np.zeros(n + m)
    z[:n] = c

    # Perform the simplex algorithm until an optimal solution is found or the problem is infeasible or unbounded
    while True:
        # Find the pivot column
        pivot_col = find_pivot_column(z, epsilon)

        # If no pivot column is found, then the current solution is optimal
        if pivot_col == -1:
            break

        # Find the pivot row
        pivot_row = find_pivot_row(tableau, b, pivot_col, epsilon)

        # If no pivot row is found, then the problem is unbounded
        if pivot_row == -1:
            return np.array([])  # return an empty array to indicate unboundedness

        # Perform the pivot operation
        pivot(tableau, b, z, pivot_row, pivot_col)

    # Extract the optimal solution from the final tableau
    x = np.zeros(n)
    for j in range(n):
        basic = True
        basic_row = -1
        for i in range(m):
            if tableau[i, j] == 1:
                if basic_row == -1:
                    basic_row = i
                else:
                    basic = False
                    break
            elif tableau[i, j] != 0:
                basic = False
                break
        if basic and basic_row != -1:
            x[j] = b[basic_row]
        else:
            x[j] = 0

    return x

# Main function to test the simplex method
if __name__ == "__main__":
    # Define the input data for a linear programming problem
    c = np.array([1, 1])  # coefficients of objective function (maximize z = 3x + 2y)
    a = np.array([[1, 1], [1, 1]])  # coefficients of constraint functions (x + y <= 4, 2x + y <= 5, y <= 3)
    b = np.array([1, 1])  # right-hand side numbers of constraint functions

    epsilon = 1e-6  # approximation accuracy

    # Call the simplex method to solve the problem
    x = simplex(a, b, c, epsilon)

    # Print the output
    if x.size == 0:  # if the problem is unbounded
        print("The method is not applicable!")
    else:
        print("The optimal solution is:")
        print_vector(x)

        z = np.dot(c, x)  # compute the objective function value
        print(f"The maximum value of the objective function is: {z}")
