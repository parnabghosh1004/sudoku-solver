def notInRow(arr, row):  
  
    # Set to store characters seen so far.  
    st = set()  
  
    for i in range(0, 9):  
  
        # If already encountered before,  
        # return false  
        if arr[row][i] in st:  
            return False
  
        # If it is not an empty cell, insert value  
        # at the current cell in the set  
        if arr[row][i] != 0:  
            st.add(arr[row][i])  
      
    return True
  
# Checks whether there is any  
# duplicate in current column or not.  
def notInCol(arr, col):  
  
    st = set()  
  
    for i in range(0, 9):  
  
        # If already encountered before, 
        # return false  
        if arr[i][col] in st: 
            return False
  
        # If it is not an empty cell, insert  
        # value at the current cell in the set  
        if arr[i][col] != 0:  
            st.add(arr[i][col])  
      
    return True
  
# Checks whether there is any duplicate  
# in current 3x3 box or not.  
def notInBox(arr, startRow, startCol):  
  
    st = set()  
  
    for row in range(0, 3):  
        for col in range(0, 3):  
            curr = arr[row + startRow][col + startCol]  
  
            # If already encountered before,  
            # return false  
            if curr in st:  
                return False
  
            # If it is not an empty cell,  
            # insert value at current cell in set  
            if curr != 0:  
                st.add(curr)  
    return True
  
# Checks whether current row and current  
# column and current 3x3 box is valid or not  
def isValid(arr, row, col):  
  
    return (notInRow(arr, row) and notInCol(arr, col) and
            notInBox(arr, row - row % 3, col - col % 3))  

  
def isValidConfig(arr):  
  
    for i in range(9):  
        for j in range(9):  
  
            # If current row or current column or  
            # current 3x3 box is not valid, return false  
            if not isValid(arr, i, j):  
                return False
    return True

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if(board[i][j] == 0):
                return (i, j)
    return None


def is_valid(board, num, pos):
    # row
    for i in range(len(board[0])):
        if(board[pos[0]][i] == num and pos[1] != i):
            return False
    # col
    for i in range(len(board)):
        if(board[i][pos[1]] == num and pos[0] != i):
            return False
    # 3x3 cube
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, (box_y+1)*3):
        for j in range(box_x*3, (box_x+1)*3):
            if(board[i][j] == num and (i, j) != pos):
                return False
    return True 

def solve_board(board):
    if not isValidConfig(board):
        return False
    find = find_empty(board)
    if not find:
        return True
    else:
        (row, col) = find
    for i in range(1, 10):
        if is_valid(board, i, (row, col)):
            board[row][col] = i

            if(solve_board(board)):
                return True
            board[row][col] = 0
    return False
