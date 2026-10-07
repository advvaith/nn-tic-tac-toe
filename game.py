from typing import Literal


class TicTacToe:
    def __init__(self):
        self.board = [0] * 9
        # O = -1, X = 1
        self.player = -1

    def reset(self):
        self.board = [0] * 9
        self.player = -1
        

    def get_valid_moves(self, board=None) -> list[int]:
        if board is None:
            board = self.board
        return [i for i, e in enumerate(board) if e == 0]

    def make_move(self, cell: int) -> list[int]:
        if cell not in self.get_valid_moves():
            raise ValueError(f"Illegal move: {cell}")

        self.board[cell] = self.player
        self.player *= -1
        return self.board

    def check_winner(self, board=None) -> int:
        WIN_LINES = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # cols
            (0, 4, 8),
            (2, 4, 6),  # diagonals
        ]
        
        if board is None:
            board = self.board
        
        for x, y, z in WIN_LINES:
            total = board[x] + board[y] + board[z]
            
            if total == 3:
                return 1
            if total == -3:
                return -1
            
        return 0
    
    def is_done(self) -> bool:
        return self.check_winner() != 0 or not self.get_valid_moves()
