import numpy as np


def print_vector(v):
    """
    Function to print a vector.
    """
    print(" ".join(map(str, v)))


def print_matrix(m):
    """
    Function to print a matrix.
    """
    for row in m:
        print_vector(row)
    print()


def find_pivot_column(c, epsilon):
    """
    Find the pivot column in the simplex tableau.
    """
    pivot_col = -1
    min_c = 0
    for j in range(len(c)):
        if c[j] < min_c - epsilon:
            min_c = c[j]
            pivot_col = j
    return pivot_col


def find_pivot_row(a, b, pivot_col, epsilon):
    """
    Find the pivot row in the simplex tableau.
    """
    pivot_row = -1
    min_ratio = float("inf")
    for i in range(len(a)):
        if a[i][pivot_col] > epsilon:
            ratio = b[i] / a[i][pivot_col]
            if ratio < min_ratio - epsilon:
                min_ratio = ratio
                pivot_row = i
    return pivot_row


def pivot(a, b, c, pivot_row, pivot_col):
    """
    Perform the pivot operation on the simplex tableau.
    """
    m, n = a.shape
    pivot_element = a[pivot_row, pivot_col]
    a[pivot_row, :] /= pivot_element
    b[pivot_row] /= pivot_element
    for i in range(m):
        if i != pivot_row:
            factor = a[i, pivot_col]
            a[i, :] -= factor * a[pivot_row, :]
            b[i] -= factor * b[pivot_row]
    factor = c[pivot_col]
    c -= factor * a[pivot_row, :]


def simplex(a, b, c, epsilon):
    """
    Implement the simplex method for linear programming problems.
    """
    m, n = a.shape

    if np.all(a == 0) and np.all(b == 0) and np.all(c == 0):
        return []

    if len(c) != n:
        return []

    if len(b) != m:
        return []

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

    while True:
        pivot_col = find_pivot_column(z, epsilon)
        if pivot_col == -1:
            break
        pivot_row = find_pivot_row(tableau, b, pivot_col, epsilon)
        if pivot_row == -1:
            return []
        pivot(tableau, b, z, pivot_row, pivot_col)

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


if __name__ == "__main__":
    c = np.array(list(map(float, input("Enter the coefficients of the objective function: ").split())))
    m, n = map(int, input("Enter the size of matrix A (Example: 3 4): ").split())
    print("Enter the coefficients of constraint functions:")
    a = np.array([list(map(float, input().split())) for _ in range(m)])
    b = np.array(list(map(float, input("Enter the right-hand side numbers of constraint functions: ").split())))
    epsilon = float(input("Enter the approximation accuracy: "))

    x = simplex(a, b, c, epsilon)

    if len(x) == 0:
        print("The method is not applicable!")
    else:
        print("The optimal solution is:")
        print_vector(x)
        z = np.dot(c, x)
        print("The minimum value of the objective function is:", z)

