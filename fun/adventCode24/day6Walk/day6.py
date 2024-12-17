

with open('input.txt', 'r') as text:
    inputList = text.read().splitlines()
    grid = [list(line) for line in inputList]

    # directions = [
    #     (0, 1),
    #     (0, -1),
    #     (1, 0),
    #     (-1, 0),
    #     (1, 1),
    #     (-1, -1),
    #     (1, -1),
    #     (-1, 1)
    # ]

    def find(element, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == element:
                    return (i, j)
    
    startPos = find('^', grid)

    def move_count(pos, direction):
        in_the_grid = True
        uniquePositions = set()
        uniquePositions.add(pos)
        loopingPositions = set()
        while in_the_grid == True:
            if direction == 'up':
                if pos[0]-1 < 0:
                    in_the_grid = False
                elif grid[pos[0]-1][pos[1]] != '#':
                    pos = (pos[0]-1, pos[1])
                    if pos in uniquePositions:
                        loopingPositions.add(pos)
                    uniquePositions.add(pos)
                else:
                    direction = 'right'
                
            if direction == 'right':
                if pos[1]+1 >= len(grid[0]):
                    in_the_grid = False
                elif grid[pos[0]][pos[1]+1] != '#':
                    pos = (pos[0], pos[1]+1)
                    if pos in uniquePositions:
                        loopingPositions.add(pos)
                    uniquePositions.add(pos)
                else:
                    direction = 'down'
            
            if direction == 'down':
                if pos[0]+1 >= len(grid):
                    in_the_grid = False
                elif grid[pos[0]+1][pos[1]] != '#':
                    pos = (pos[0]+1, pos[1])
                    if pos in uniquePositions:
                        loopingPositions.add(pos)
                    uniquePositions.add(pos)
                else:
                    direction = 'left'
            
            if direction == 'left':
                if pos[1]-1 < 0:
                    in_the_grid = False
                elif grid[pos[0]][pos[1]-1] != '#':
                    pos = (pos[0], pos[1]-1)
                    if pos in uniquePositions:
                        loopingPositions.add(pos)
                    uniquePositions.add(pos)
                else:
                    direction = 'up'
        
        return len(loopingPositions)

    answer = move_count(startPos, 'up')
    print(answer)


    

        

        

            




 