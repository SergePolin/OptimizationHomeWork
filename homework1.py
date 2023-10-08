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
    min_c = 0  # initialize with 0 to find the most negative coefficient
    for j in range(len(c)):
        if c[j] < min_c - epsilon:  # use epsilon to avoid numerical errors
            min_c = c[j]
            pivot_col = j
    return pivot_col

# A function to find the pivot row in the simplex tableau
def find_pivot_row(a, b, pivot_col, epsilon):
    pivot_row = -1  # initialize with -1 to indicate no pivot row
    min_ratio = float("inf")  # initialize with infinity to find the minimum ratio
    for i in range(len(a)):
        if a[i][pivot_col] > epsilon:  # use epsilon to avoid division by zero
            ratio = b[i] / a[i][pivot_col]  # compute the ratio of b[i] to a[i][pivot_col]
            if ratio < min_ratio - epsilon:  # use epsilon to avoid numerical errors
                min_ratio = ratio
                pivot_row = i
    return pivot_row

# A function to perform the pivot operation on the simplex tableau
def pivot(a, b, c, pivot_row, pivot_col):
    m, n = a.shape

    # divide the pivot row by the pivot element
    pivot_element = a[pivot_row, pivot_col]
    a[pivot_row, :] /= pivot_element
    b[pivot_row] /= pivot_element

    # subtract multiples of the pivot row from other rows
    for i in range(m):
        if i != pivot_row:
            factor = a[i, pivot_col]
            a[i, :] -= factor * a[pivot_row, :]
            b[i] -= factor * b[pivot_row]

    # update the objective function coefficients
    factor = c[pivot_col]
    c -= factor * a[pivot_row, :]

# A function to implement the simplex method for linear programming problems
def simplex(a, b, c, epsilon):
    m, n = a.shape

    # check if the count of elements in C is equal to the count of columns in A
    if len(c) != n:
        return []

    # check if the count of elements in b is equal to the count of rows in A
    if len(b) != m:
        return []

    # create an initial feasible solution by adding slack variables
    tableau = np.zeros((m, n + m))
    for i in range(m):
        for j in range(n + m):
            if j < n:
                tableau[i, j] = a[i, j]
            elif j == n + i:
                tableau[i, j] = 1
            else:
                tableau[i, j] = 0

    z = np.zeros(n + m)
    z[:n] = c

    # perform the simplex algorithm until an optimal solution is found or the problem is infeasible or unbounded
    while True:
        # find the pivot column
        pivot_col = find_pivot_column(z, epsilon)

        # if no pivot column is found, then the current solution is optimal
        if pivot_col == -1:
            break

        # find the pivot row
        pivot_row = find_pivot_row(tableau, b, pivot_col, epsilon)

        # if no pivot row is found, then the problem is unbounded
        if pivot_row == -1:
            return []

        # perform the pivot operation
        pivot(tableau, b, z, pivot_row, pivot_col)

    # extract the optimal solution from the final tableau
    x = np.zeros(n)
    for j in range(n):
        basic = False
        basic_row = -1
        for i in range(m):
            if tableau[i, j] == 1:
                if basic_row == -1:
                    basic_row = i
                    basic = True
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

# A main function to test the simplex method
if __name__ == "__main__":
    # define the input data for a linear programming problem
    c = np.array([2,2,1], dtype=float)  # coefficients of objective function 
    a = np.array([[1, 1, 1], [2, -1, 1]], dtype=float)  # coefficients of constraint functions
    b = np.array([4], dtype=float)  # right-hand side numbers of constraint functions

    epsilon = 1e-6  # approximation accuracy

    # call the simplex method to solve the problem
    x = simplex(a, b, c, epsilon)

    # print the output
    if len(x) == 0:  # if the problem is unbounded
        print("The method is not applicable!")
    else:  # otherwise, print the optimal solution and the objective function value
        print("The optimal solution is:")
        print_vector(x)

        z = np.dot(c, x)  # compute the objective function value
        print("The minimum value of the objective function is:", z)

