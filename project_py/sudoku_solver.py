import numpy as np

SIZE = 9
matrix = []
x_list = []

#function to print sudoku
def print_sudoku():
    # for i in matrix:
    #     print (i)
    sudo = []
    
    for i in range(len(matrix[:])):
        string = ""
        for j in range(len(matrix[i])):
            if x_list[i][j] != "0":
                string += "0"
            else:
                string += str(matrix[i][j])
            # print(matrix)

        sudo.append(string)
    # print(sudo)
    return sudo


#function to check if all cells are assigned or not
#if there is any unassigned cell
#then this function will change the values of
#row and col accordingly
def number_unassigned(row, col):
    num_unassign = 0
    for i in range(0,SIZE):
        for j in range (0,SIZE):
            #cell is unassigned
            if matrix[i][j] == 0:
                row = i
                col = j
                num_unassign = 1
                a = [row, col, num_unassign]
                return a
    a = [-1, -1, num_unassign]
    return a
#function to check if we can put a
#value in a paticular cell or not
def is_safe(n, r, c):
    #checking in row
    for i in range(0,SIZE):
        #there is a cell with same value
        if matrix[r][i] == n:
            return False
    #checking in column
    for i in range(0,SIZE):
        #there is a cell with same value
        if matrix[i][c] == n:
            return False
    row_start = (r//3)*3
    col_start = (c//3)*3;
    #checking submatrix
    for i in range(row_start,row_start+3):
        for j in range(col_start,col_start+3):
            if matrix[i][j]==n:
                return False
    return True

#function to check if we can put a
#value in a paticular cell or not
def solve_sudoku():
    row = 0
    col = 0
    #if all cells are assigned then the sudoku is already solved
    #pass by reference because number_unassigned will change the values of row and col
    a = number_unassigned(row, col)
    if a[2] == 0:
        return True
    row = a[0]
    col = a[1]
    #number between 1 to 9
    for i in range(1,10):
        #if we can assign i to the cell or not
        #the cell is matrix[row][col]
        if is_safe(i, row, col):
            matrix[row][col] = i
            #backtracking
            if solve_sudoku():
                return True
            #f we can't proceed with this solution
            #reassign the cell
            matrix[row][col]=0
    return False

def sudoku_pro(x):
    global matrix
    global x_list
    x_list = x
    # x="85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4."
    

    ###### 포맷 변경
    met = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == "0":
                met.append(int(0))
            else: met.append(int(x[i][j]))


    met = np.array(met).reshape((9,9))
    matrix = met.tolist()    
    
    if solve_sudoku():
        return print_sudoku()


    else:
        print("No solution")