# OptimizationHomeWork

## How to run

```bash
git clone https://github.com/SergePolin/OptimizationHomeWork
cd OptimizationHomeWork
python homework1.py
```

## Example

```
Enter coefficients of the objective funciton, separated by space (for example "1 2 3"):
1 2 3

Enter the number of constraints functions: 3
Enter coefficients (3) of the constraint funciton №1:
-1 2 5
Enter coefficients (3) of the constraint funciton №2:
5 2 6
Enter coefficients (3) of the constraint funciton №3:
3 1 7

Enter b vector (size = 3):
0 0 0

----------------------------

Objective function coefs: [1.0, 2.0, 3.0]
Constraint functions coefs:
 [[-1.  2.  5.]
 [ 5.  2.  6.]
 [ 3.  1.  7.]]
B vector: [0.0, 0.0, 0.0]
Accuracy: 1e-06

----------------------------

Optimal solution: [0. 0. 0.]
Optimal value: 0.0

----------------------------
```
