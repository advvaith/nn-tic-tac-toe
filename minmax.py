from game import TicTacToe

transposition = {}
game = TicTacToe()

def terminal_score(board: list) -> int | None:
    winner = game.check_winner(board)
    if winner == -1 or winner == 1:
        return winner
    else:
        if 0 in board:
            return None
        else:
            return 0

def best_score(board: list, player: int) -> int:
    score = terminal_score(board)
    if score is not None:
        return score
    
    scores = []
    for i, e in enumerate(board):
        if e == 0:
            board_cpy = board[:]
            board_cpy[i] = player
            next_move = tuple(board_cpy),-player
            if (next_move) in transposition:
                scores.append(transposition[next_move])
            else:
                min_move = best_score(board_cpy, -player)
                transposition[next_move] = min_move
                scores.append(min_move)
            
    return max(scores) if player == 1 else min(scores)

def choose_move(board, player):
    best_cell = None
    best_val = None
    
    for i, e in enumerate(board):
        if e == 0:
            board_cpy = board[:]
            board_cpy[i] = player
            score = best_score(board_cpy, -player)
            
            if best_val is None or (player == 1 and score > best_val) or (player == -1 and score < best_val):
                best_val = score
                best_cell = i
                
    return best_cell
        
if __name__ == "__main__":
    game.make_move(0)
    assert terminal_score(game.board) == None
    game.make_move(3)
    game.make_move(1)
    game.make_move(4)
    game.make_move(2)
    assert terminal_score(game.board) == -1
    
    assert best_score([0] * 9, 1) == 0
    assert best_score([1, 1, 0, 0, -1, 0, 0, 0, 0], 1) == 1
    assert best_score([1, 0, 0, 0, -1, -1, 0, 1, 0], -1) == -1
    assert best_score([1, 0, 1, 0, -1, 0, 0, 0, 0], -1) == 0
    
    assert choose_move([1, 1, 0, 0, -1, 0, 0, 0, 0], 1) == 2
    