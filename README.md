# OptimizationHomeWork
[![Run pytest](https://github.com/SergePolin/OptimizationHomeWork/actions/workflows/main.yml/badge.svg)](https://github.com/SergePolin/OptimizationHomeWork/actions/workflows/main.yml)
This repository contains a Python implementation of an optimization algorithm for solving linear programming problems. The algorithm uses the simplex method to find the optimal solution to a given linear programming problem.

## Dependencies

To run this program, you will need to have the following dependencies installed:

- Python 3.x
- NumPy

## How to run

```bash
git clone https://github.com/SergePolin/OptimizationHomeWork
cd OptimizationHomeWork
pip install -r requirements.txt
python homework1.py
```

## Example

```
Enter the coefficients of the objective function: 2 3 -1
Enter the size of matrix A (Example: 3 4): 2 3
Enter the coefficients of constraint functions:
1 1 1
2 -1 1
Enter the right-hand side numbers of constraint functions: 4 2
Enter the approximation accuracy: 1e-6
The optimal solution is:
0.0 0.0 2.0
The minimum value of the objective function is: -2.0
```
