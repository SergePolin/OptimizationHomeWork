# OptimizationHomeWork

[![Run pytest](https://github.com/SergePolin/OptimizationHomeWork/actions/workflows/main.yml/badge.svg)](https://github.com/SergePolin/OptimizationHomeWork/actions/workflows/main.yml)

This repository contains a Python implementation of an optimization algorithm for solving linear programming problems. The algorithm uses the simplex method to find the optimal solution to a given linear programming problem.

## Dependencies

To run this program, you will need to have the following dependencies installed:

- Python 3.x
- NumPy
- Pytest

## How to run

```bash
git clone https://github.com/SergePolin/OptimizationHomeWork
cd OptimizationHomeWork
pip install -r requirements.txt
python homework1.py
# or
python homework2.py
# or
python homework3.py
```

# How to run tests

```bash
pytest tests.py
```

## Example

### Homework 1 & 2

```bash
# input
Enter the coefficients of the objective function: 2 3 -1
Enter the size of matrix A (Example: 3 4): 2 3
Enter the coefficients of constraint functions:
1 1 1
2 -1 1
Enter the right-hand side numbers of constraint functions: 4 2
Enter the approximation accuracy: 1e-6
# output
The optimal solution is:
0.0 0.0 2.0
The minimum value of the objective function is: -2.0
```

### Homework 3

```bash
Enter vector of coefficients of supply: 10 15
Enter the size of matrix C (Example: 3 4): 2 3
Enter matrix of coefficients of costs:
6 7 8
15 80 78
Enter vector of coefficients of demand: 15 5 5
-------------------
Vector S
[10. 15.]
Matrix C
[[ 6.  7.  8.]
 [15. 80. 78.]]
Vector D
[15.  5.  5.]
-------------------
North-West method:
The initial basic feasible solution is:
[10.0, 0.0, 0.0, 5.0, 5.0, 5.0]
Vogel's approximation method:
The initial basic feasible solution is:
[0, 5, 5, 15, 0, 0]
Russell's approximation method:
The initial basic feasible solution is:
[0, 5, 5, 15, 0, 0]
```
