with open('input.txt', 'r') as text:
    input_list = text.read().splitlines()
    words = 0

    # Count horizontal occurrences
    for line in input_list:
        words += line.count('XMAS')
        words += line.count('SAMX')

    # Count vertical occurrences (transpose and analyze)
    transposed = [''.join(column) for column in zip(*input_list)]
    for line in transposed:
        words += line.count('XMAS')
        words += line.count('SAMX')

    # Helper function to extract diagonals from a grid
    def get_diagonals(grid):
        diagonals = []
        rows = len(grid)
        cols = len(grid[0])

        # Top-left to bottom-right (\)
        for d in range(rows + cols - 1):
            diag = []
            for i in range(max(0, d - cols + 1), min(rows, d + 1)):
                j = d - i
                diag.append(grid[i][j])
            diagonals.append(''.join(diag))

        # Top-right to bottom-left (/)
        for d in range(rows + cols - 1):
            diag = []
            for i in range(max(0, d - cols + 1), min(rows, d + 1)):
                j = cols - 1 - (d - i)
                diag.append(grid[i][j])
            diagonals.append(''.join(diag))

        return diagonals

    # Convert input list into a character grid
    char_grid = [list(line) for line in input_list]

    # Get all diagonals
    diagonals = get_diagonals(char_grid)

    # Count diagonal occurrences
    for line in diagonals:
        words += line.count('XMAS')
        words += line.count('SAMX')

    print(words)
