

with open('input.txt', 'r') as text:

    inputList = text.read().splitlines()
    # words = 0
    # vertical = []
    # for i in inputList:
    #     words = words + i.count('XMAS')
    #     words = words + i.count('SAMX')
    

    # vertical = [''.join(column) for column in zip(*inputList)]
    # for j in vertical:
    #     words = words + j.count('XMAS')
    #     words = words + j.count('SAMX')
    
    
    # 
    # cols = len(char_grid[0])
    # print(cols)
    # count = 0
    # grid = [list(line) for line in inputList]
    # for i in range (1, len(grid)-1):
    #     for j in range(1, len(grid[0])-1):
    #         if grid[i][j] == 'A':
    #             if grid[i - 1][j - 1] == 'M' and grid[i - 1][j + 1] == 'S':
    #                 if grid[i + 1][j - 1] == 'M' and grid[i + 1][j + 1] == 'S':
    #                     count += 1
    #             if grid[i - 1][j - 1] == 'S' and grid[i - 1][j + 1] == 'S':
    #                 if grid[i + 1][j - 1] == 'M' and grid[i + 1][j + 1] == 'M':
    #                     count += 1
    #             if grid[i - 1][j - 1] == 'S' and grid[i - 1][j + 1] == 'M':
    #                 if grid[i + 1][j - 1] == 'S' and grid[i + 1][j + 1] == 'M':
    #                     count += 1
    #             if grid[i - 1][j - 1] == 'M' and grid[i - 1][j + 1] == 'M':
    #                 if grid[i + 1][j - 1] == 'S' and grid[i + 1][j + 1] == 'S':
    #                     count += 1
    
    # print(count)




    # count = 0
    # # Directions: (dy, dx) for vertical, horizontal, diagonal, and their reverses
    # directions = [
    #     (0, 1),  # right (horizontal)
    #     (0, -1),  # left (horizontal)
    #     (1, 0),  # down (vertical)
    #     (-1, 0),  # up (vertical)
    #     (1, 1),  # diagonal down-right
    #     (-1, -1),  # diagonal up-left
    #     (1, -1),  # diagonal down-left
    #     (-1, 1),  # diagonal up-right
    # ]
    # grid = [list(line) for line in inputList]
    # for i in range (len(grid)):
    #     for j in range(len(grid[0])):
    #         if grid[i][j] == 'X':
    #             # Check all directions
    #             for dy, dx in directions:
    #                 # Check if we can go in the given direction and stay within bounds
    #                 if 0 <= i + 3 * dy < len(grid) and 0 <= j + 3 * dx < len(grid[0]):
    #                     # Collect the 4 characters in the direction
    #                     chars = [grid[i + k * dy][j + k * dx] for k in range(4)]
    #                     # Check if the characters match "XMAS" or "SAMX"
    #                     if chars == ['X', 'M', 'A', 'S'] or chars == ['S', 'A', 'M', 'X']:
    #                         count += 1
    # print(count)