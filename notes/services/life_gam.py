import random
import os
import time

class GameOfLife:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cots = cols
        self.grid = []

        for i in range(self.row):
            row = []
            for j in range(self.cols):
                if random.randint(0, 1) == 0:
                    row.append(False) # Dead cell
                else:
                    row.append(True) # Live cell
            self.grid.append(row)

    def print_grid(self):
        for i in range(self.rows):
            for j in range(self.cots):
                if self.grid[i][j] is True:
                    print('■', end=' ')
                else:
                    print('□', end=' ')
            print('\n')

    def update_grid(self):
        new_grid = []
        for i in range(self.row):
            row = []
            for j in range(self.cols):
                current_cell = self.grid[i][j]
                live_neighbours = 0

                # Check all 8 neighbouring cells
                for row_offset in range(-1, 2):
                    for col_offset in range(-1, 2):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        row_index = i + row_offset
                        col_index = j + col_offset
                        if (row_index >= 0 and row_index < self.rows and
                           col_index >= 0 and col_index < self.cots and
                           self.grid[row_index][col_index] is True):
                            live_neighbours +=1
                
                # Game of Life rules
                if current_cell is True and (live_neighbours < 2 or live_neighbours > 3):
                    row.append(False)
                elif current_cell is False and live_neighbours == 3:
                    row.append(True)
                else: # Simple survival or empty cell
                    row.append(current_cell)
            
            new_grid.append(row)
        
        self.grid = new_grid
    
if __name__ == "__main__":
    game = GameOfLife(10, 10)
    
    while True:
        os.system('clear')
        game.print_grid()
        game.update_grid()
        time.sleep(0.5)