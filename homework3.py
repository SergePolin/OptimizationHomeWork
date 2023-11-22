import numpy as np

# define infinity large value
M = 10**10

# Funtion to check if the problem is balanced
def check_balance(S, C, D):
    if sum(S) != sum(D):
        print("The problem is not balanced!")
        return False


# Function to check if the method is applicable
def check_applicable(S, C, D):
    if len(S) != len(C) or len(D) != len(C[0]):
        print("The method is not applicable!")
        return False


# Vogel's approximation method
def vogel(costs, supply, demand):
    INF = 10 ** 10
    n, m = costs.shape

    initial_solution = np.zeros((n, m), dtype=int)

    # check if the problem is balanced
    check_balance(supply, costs, demand)

    # check if the method is applicable
    check_applicable(supply, costs, demand)


    def find_diff(costs):
        row_diff = np.array([np.partition(row, 1)[1] - np.partition(row, 0)[0] for row in costs])
        col_diff = np.array([np.partition(costs[:, col], 1)[1] - np.partition(costs[:, col], 0)[0] for col in range(m)])
        return row_diff, col_diff

    while np.max(supply) != 0 or np.max(demand) != 0:
        row_diff, col_diff = find_diff(costs)
        maxi1 = np.max(row_diff)
        maxi2 = np.max(col_diff)

        if maxi1 >= maxi2:
            for ind, val in enumerate(row_diff):
                if val == maxi1:
                    mini1 = np.min(costs[ind])
                    ind2 = np.argmin(costs[ind])
                    mini2 = min(supply[ind], demand[ind2])
                    supply[ind] -= mini2
                    demand[ind2] -= mini2
                    initial_solution[ind][ind2] = mini2
                    if demand[ind2] == 0:
                        costs[:, ind2] = INF
                    else:
                        costs[ind] = INF
                    break
        else:
            for ind, val in enumerate(col_diff):
                if val == maxi2:
                    mini1 = INF
                    for j in range(n):
                        mini1 = min(mini1, costs[j, ind])
                    ind2 = np.argmin(costs[:, ind])
                    mini2 = min(supply[ind2], demand[ind])
                    supply[ind2] -= mini2
                    demand[ind] -= mini2
                    initial_solution[ind2][ind] = mini2
                    if demand[ind] == 0:
                        costs[:, ind] = INF
                    else:
                        costs[ind2] = INF
                    break

    return [j for i in initial_solution for j in i]

def russell(costs, supply, demand):
    costs = np.array(costs)
    supply = np.array(supply)
    demand = np.array(demand)
    initial_solution = np.zeros_like(costs)

    # check if the problem is balanced
    check_balance(supply, costs, demand)

    # check if the method is applicable
    check_applicable(supply, costs, demand)

    while np.any(supply) and np.any(demand):
        max_in_rows = np.max(costs * (supply[:, None] > 0), axis=1)
        max_in_columns = np.max(costs * (demand > 0), axis=0)

        new_table = costs - max_in_rows[:, None] - max_in_columns

        x_coord, y_coord = np.unravel_index(np.argmin(new_table, axis=None), new_table.shape)
        value = min(supply[x_coord], demand[y_coord])
        initial_solution[x_coord, y_coord] = value

        supply[x_coord] -= value
        demand[y_coord] -= value

    return [j for i in initial_solution for j in i]

def north_west(S, C, D):
    
    # check if the problem is balanced
    check_balance(S, C, D)
    # check if the method is applicable
    check_applicable(S, C, D)

    x0 = np.zeros_like(C)
    i, j = 0, 0

    while i < len(S) and j < len(D):
        allocation = min(S[i], D[j])
        x0[i][j] = allocation
        S[i] -= allocation
        D[j] -= allocation

        if S[i] == 0 and i < len(S) - 1:
            i += 1
        elif D[j] == 0 and j < len(D) - 1:
            j += 1
        elif S[i] == 0 and D[j] == 0:
            break
        else:
            continue
    return [j for i in x0 for j in i]


if __name__ == "__main__":
    s = np.array(list(map(float, input("Enter vector of coefficients of supply: ").split())))
    m, n = map(int, input("Enter the size of matrix C (Example: 3 4): ").split())
    print("Enter matrix of coefficients of costs:")
    c = np.array([list(map(float, input().split())) for _ in range(m)])
    d = np.array(list(map(float, input("Enter vector of coefficients of demand: ").split())))
    print('-------------------')
    print('Vector S')
    print(s)
    print('Matrix C')
    print(c)
    print('Vector D')
    print(d)
    print('-------------------')
    print("North-West method:")
    print("The initial basic feasible solution is:")

    # copy
    print(north_west(s.copy(), c.copy(), d.copy()))
    print("Vogel's approximation method:")
    print("The initial basic feasible solution is:")
    print(vogel(c.copy(), s.copy(), d.copy()))
    print("Russell's approximation method:")
    print("The initial basic feasible solution is:")
    print(russell(c.copy(), s.copy(), d.copy()))
    

    
    

