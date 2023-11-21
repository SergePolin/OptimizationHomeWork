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
def vogel(S, C, D):
    x0 = np.array([[0]*len(D)]*len(S)) # List to store the initial basic feasible solution
    index_r, index_c = -1, -1 # The index of row and column that will be changed

    # Check if the problem is balanced and method is aplicable
    check_applicable(S, C, D)
    check_balance(S, C, D)

    # Function to find difference between two min elements in each row
    def min_dif_row(C):
        mind_row = [0]*len(S)
        for i in range(len(C)):
            if sorted(C[i])[1] != M:
                mind_row[i] += abs(sorted(C[i])[0] - sorted(C[i])[1])
        return mind_row
    
    # Function to find difference between two min elements in each column
    def min_dif_col(C):
        mind_col = [0]*len(D)
        for i in range(len(C.T)):
            if sorted(C.T[i])[1] != M:
                mind_col[i] += abs(sorted(C.T[i])[0] - sorted(C.T[i])[1])
        return mind_col
    
    # Function to set M-s
    def set_M(C, r_c, index): # Matrix C, row or column, index of r_c
        if r_c == "c":
            C = C.transpose()
            C[index] = [M]*len(C[index])
            C = C.transpose()
        elif r_c == "r":
            C[index] = [M]*len(C[index])
    
    # Main loop
    while True:
        index_c, index_r = -1, -1
        # Check if the algorithm is still applicable
        flag_to_break = True
        for i in range(len(S)):
            if S[i] != 0:
                flag_to_break = False
                break
        if flag_to_break:
            break

        # Step 1 - Find difference between two min elements in each row and column
        mind_row = min_dif_row(C)
        mind_col = min_dif_col(C)

        # Step 2 - Determine the row or column to manipulate
        max_md_r = max(mind_row)
        max_md_c = max(mind_col)
        if max_md_r > max_md_c:
            index_r = mind_row.index(max_md_r)
        else:
            index_c = mind_col.index(max_md_c)
        
        if index_r != -1:
            index_c = list(C[index_r]).index(min(C[index_r]))
        else:
            index_r = list(C.transpose()[index_c]).index(min(C.transpose()[index_c]))

        x0[index_r][index_c] = min(S[index_r], D[index_c])

        # Step 3 - Change S and D
        S[index_r] -= x0[index_r][index_c]
        D[index_c] -= x0[index_r][index_c]

        # Step 4 - Set M-s in column or in row of C
        if S[index_r] == 0:
            set_M(C, "r", index_r)
        else:
            set_M(C, "c", index_c)

        # Repeat
    return [j for i in x0 for j in i]

def russel(S, C, D):
    x0 = np.array([[0]*len(D)]*len(S)) # List to store the initial basic feasible solution
    index_r, index_c = -1, -1 # The index of row and column that will be changed

    # Check if the problem is balanced and method is aplicable
    check_applicable(S, C, D)
    check_balance(S, C, D)

    # Function to find the max element in each row
    def max_row(C):
        m_r = [0]*len(S)
        for i in range(len(C)):
            m_r[i] = max(C[i])
        return m_r

    # Function to find the max element in each column
    def max_col(C):
        m_c = [0]*len(D)
        for i in range(len(C.T)):
            m_c[i] = max(C.T[i])
        return m_c
    
    # Function to calculate new matrix C_d there C_d[i][j] = C[i][j] - m_r[i] - m_c[j]
    def new_C_d(C, m_r, m_c):
        C_d = np.zeros((len(C), len(C[0])))
        for i in range(len(C)):
            for j in range(len(C[0])):
                C_d[i][j] = C[i][j] - m_r[i] - m_c[j]
        return C_d

    
    # Function to set M-s
    def set_M(C, r_c, index): # Matrix C, row or column, index of r_c
        if r_c == "c":
            C = C.transpose()
            C[index] = [M]*len(C[index])
            C = C.transpose()
        elif r_c == "r":
            C[index] = [M]*len(C[index])
    

    # Creat lists of max values for each row and column
    max_row_val = max_row(C)
    max_col_val = max_col(C)

    # Create new matrix C_d
    C_d = new_C_d(C, max_row_val, max_col_val)
    
    # Main loop
    while True:
        index_c, index_r = -1, -1
        # Check if the algorithm is still applicable
        flag_to_break = True
        for i in range(len(S)):
            if S[i] != 0:
                flag_to_break = False
                break
        if flag_to_break:
            break

        # Step 1 - Find the most negative element in the C_d and his index
        min_temp = [M, C.max()]
        for i in range(len(C_d)):
            for j in range(len(C_d[0])):
                if C_d[i][j] == min_temp[0]:
                    if C[i][j] < min_temp[1]:
                        min_temp = [C_d[i][j], C_d[i][j]]
                        index_r = i
                        index_c = j
                elif C_d[i][j] < min_temp[0]:
                    min_temp = [C_d[i][j], C[i][j]]
                    index_r = i
                    index_c = j
                
        x0[index_r][index_c] = min(S[index_r], D[index_c]) # Store the basic feasible solution

        # Step 2 - Change S and D
        S[index_r] -= x0[index_r][index_c]
        D[index_c] -= x0[index_r][index_c]

        # Step 3 - Set M-s
        if S[index_r] == 0:
            set_M(C_d, "r", index_r)
        else:
            set_M(C_d, "c", index_c)
            
        # Repeat
    return [j for i in x0 for j in i]

def north_west(S, C, D):
    if len(S) != C.shape[0] or len(D) != C.shape[1]:
        return "The method is not applicable!"

    if sum(S) != sum(D):
        return "The problem is not balanced!"

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
    print(vogel(s.copy(), c.copy(), d.copy()))
    print("Russell's approximation method:")
    print("The initial basic feasible solution is:")
    print(russel(s.copy(), c.copy(), d.copy()))

    
    

