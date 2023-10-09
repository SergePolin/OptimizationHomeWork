from homework1 import *

def test_find_pivot_column():
    c = np.array([3, -2, 1])
    epsilon = 1e-6
    pivot_col = find_pivot_column(c, epsilon)
    assert pivot_col == 1

def test_simplex1():
    a = np.array([[2, 1, 1], [1, -1, 2], [3, 0, 1]])
    b = np.array([2, 2, 3])
    c = np.array([-1, 2, 1])
    epsilon = 1e-6
    res = calculateSimplex(a, b, c, epsilon)
    assert np.allclose(res[0], np.array([1, 0, 0]))
    assert np.allclose(res[1], -1)

def test_simplex2():
    a = np.array([[1, 1, 1], [2, -1, 1]])
    b = np.array([4, 2])
    c = np.array([2, 3, -1])
    epsilon = 1e-6
    res = calculateSimplex(a, b, c, epsilon)
    assert np.allclose(res[0], np.array([0, 0, 2]))
    assert np.allclose(res[1], -2)

# def test_simplex3():
#     a = np.array([[0, 0], [0, 0], [0, 0]])
#     b = np.array([0, 0])
#     c = np.array([0, 0])
#     epsilon = 1e-6
#     x, z = simplex(a, b, c, epsilon)
#     z = np.dot(c, x)
#     assert np.allclose(x, np.array([0, 0]))
#     assert np.allclose(z, 0)

def test_simplex3():
    a = np.array([[1, 1], [1, 1]])
    b = np.array([1, 1])
    c = np.array([1, 1])
    epsilon = 1e-6
    res = calculateSimplex(a, b, c, epsilon)
    assert np.allclose(res[0], np.array([0, 0]))
    assert np.allclose(res[1], 0)

def test_simplex4():
    a = np.array([[1]])
    b = np.array([2])
    c = np.array([-2])
    epsilon = 1e-6
    res = calculateSimplex(a, b, c, epsilon)
    assert np.allclose(res[0], np.array([2]))
    assert np.allclose(res[1], -4)

# def test_simplex5():
#     a = np.array([[1, -3], [-2, -1], [11, -0.5]])
#     b = np.array([4, 2, -7])
#     c = np.array([2, 3])
#     epsilon = 1e-6
#     res = calculateSimplex(a, b, c, epsilon)
#     print("teeeee", res)
#     assert np.allclose(res[0], np.array([0, 14]))
#     assert np.allclose(res[1], 42)

# def test_simplex6():
#     a = np.array([[1, -3], [-2, -1], [11, -0.5]])
#     b = np.array([4, 2, -7])
#     c = np.array([2, 3])
#     epsilon = 1e-6
#     res = calculateSimplex(a, b, c, epsilon)
#     print("teeeee", res)
#     assert np.allclose(res[0], np.array([0, 14]))
#     assert np.allclose(res[1], 42)

def test_simplex7():
    a = np.array([[-1], [2]])
    b = np.array([3, 10])
    c = np.array([-2])
    epsilon = 1e-6
    print('A', a)
    print('B', b)
    print('C', c)
    res = calculateSimplex(a, b, c, epsilon)
    assert np.allclose(res[0], np.array([5]))
    assert np.allclose(res[1], -10)



# cases with exceptions:

# (1)
# Enter the coefficients of the objective function: -2
# Enter the size of matrix A (Example: 3 4): 1 1
# Enter the coefficients of constraint functions:
# -1
# Enter the right-hand side numbers of constraint functions: 3
# Enter the approximation accuracy: 1e-6
# The method is not applicable!
# PASSED


# (2)
# Enter the coefficients of the objective function: -2
# Enter the size of matrix A (Example: 3 4): 2 1
# Enter the coefficients of constraint functions:
# -1
# 2
# Enter the right-hand side numbers of constraint functions: 3 5 6
# Enter the approximation accuracy: 1e-6
# The method is not applicable!
# PASSED
