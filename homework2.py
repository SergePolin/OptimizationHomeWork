import numpy as np
from numpy.linalg import norm

# old algorithm: simplex method

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


def calculateSimplex(a,b,c,epsilon):

    x = simplex(a, b, c, epsilon)

    if len(x) == 0:
        print("The method is not applicable!")
        return -1
    z = np.dot(c, x)
    return [x,z]

# new algorithm: interior point method

def has_interior_point(A, b):
    # An interior point exists if all elements of b are strictly greater than zero
    return np.all(b > 0)

def compute(A,B):
    return np.dot(A,B)

def calculateMatrixD(c,b):
    m = len(b)
    n = len(c)
    H = np.ones((1,m+n))
    D = np.zeros((m+n,m+n))
    for i in range(m+n):
        D[i,i] = H[0,i]
    return D


def is_unbounded(A, b, c):
    # Number of constraints
    m = A.shape[0]

    # Number of variables
    n = A.shape[1]

    # Initialize tableau
    T = np.zeros((m+1, n+m+2))
    T[:m, :n] = A
    T[:m, n:n+m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = c

    # Simplex method
    while True:
        # Find pivot column (most negative element in last row)
        pivot_col = np.argmin(T[-1, :-1])

        # If all elements in pivot column are non-positive, problem is unbounded
        if np.all(T[:-1, pivot_col] <= 0):
            return True

        # Find pivot row (smallest non-negative ratio)
        ratios = np.divide(T[:-1, -1], T[:-1, pivot_col], out=np.full_like(T[:-1, -1], np.inf), where=T[:-1, pivot_col]>0)
        pivot_row = np.argmin(ratios)

        # Pivot
        T[pivot_row] /= T[pivot_row, pivot_col]
        for i in range(m+1):
            if i != pivot_row:
                T[i] -= T[i, pivot_col] * T[pivot_row]

        # If no negative elements in last row, optimal solution found
        if np.all(T[-1, :-1] >= 0):
            return False


# function to find initial feasible trial solution of problem
def initial_trial_sol(A, b):
    # A is 1 dimensional matrix
    if A.ndim == 1:
        return np.array([b[0] / A[i] / len(A) for i in range(len(A))])
    
    try:
        A = newA(A, b)
        # Try to solve the system of equations
        x = np.dot(np.linalg.pinv(A), b)
        return x
    except:
        # If the matrix is not invertible, the system is not feasible
        print("The problem does not have solution!")
        exit()
    

# function to compute D
def compute_D(x):
    return np.diag(x)
    
# function to compute new A with slack variables
def getNewA(A, b):
    n,m = A.shape
    newA = np.zeros((n,m+len(b)))
    for i in range(n):
        for j in range(m):
            newA[i][j] = A[i][j]
        for j in range(m,m+len(b)):
            if (j-m == i):
                newA[i][j] = 1
            newA[i][j] = 0
    return newA

def getNu(cp):
    return np.absolute(np.min(cp))


# function to inialize Interior Point Algorithm
def interior_point_algo(c, A, b, alph=0.5):

    iter = 500 # value of iteration
    x = initial_trial_sol(A, b) # initial trial solution with slack variables
    s = [] # value of objective function

    # check if the system has interior point
    if not has_interior_point(A, b):
        print("The method is not applicable!")
        exit()

    # check if the system is unbounded
    if A.ndim > 1:
        if is_unbounded(A, b, c):
            print("The problem does not have solution!")
            exit()


    # update c with slack variables
    newC = np.zeros(len(c)+len(b))
    newC[0:len(c)] = c
    oldC = c
    c = newC

    # update A with slack variables
    A = getNewA(A,b)
    # main loop of iterative Interior Point Algorithm
    while (iter):
        D = compute_D(x)
        u = x
        Astar = compute(A,D)
        cc = compute(D,c)
        I = np.eye(len(c))
        F = compute(Astar, np.transpose(Astar))
        if np.linalg.det(F) == 0:
            print("The problem does not have solution!")
            print("The matrix is not invertible!")
            exit()
        FI = np.linalg.inv(F)
        H = compute(np.transpose(Astar), FI)
        J = compute(H, Astar)
        P = np.subtract(I, J)
        cp = np.dot(P,cc)
        nu = getNu(cp)
        coeff = (alph/nu)*cp
        y = np.add(np.ones(len(c),float),coeff)
        yy = np.dot(D,y)
        x = yy

        if (np.dot(A, x) <= b).all():
            s.append([np.dot(c, x).sum(), x])
        
        if norm(np.subtract(yy,u),ord = 2)< 0.0001: 
            break

        iter -= 1

    # get the final result
    x = max(s, key=lambda x: x[0])[1]
    ansX = x[0:len(c)-len(b)]
    z = np.dot(oldC, ansX)
    # print the result
    print(f"x = {ansX}")
    print(f"alpha = {alph}, z = {z}")

if __name__ == "__main__":
    c = np.array(list(map(float, input("Enter the coefficients of the objective function: ").split())))
    m, n = map(int, input("Enter the size of matrix A (Example: 3 4): ").split())
    print("Enter the coefficients of constraint functions:")
    a = np.array([list(map(float, input().split())) for _ in range(m)])
    b = np.array(list(map(float, input("Enter the right-hand side numbers of constraint functions: ").split())))
    epsilon = float(input("Enter the approximation accuracy: "))
    interior_point_algo(c, a, b, alph=0.5)
    res = calculateSimplex(a,b,c,epsilon)
    if res != -1:
        print("Simplex method:")
        print("The optimal solution is:")
        print_vector(res[0])
        print("The minimum value of the objective function is:", res[1])