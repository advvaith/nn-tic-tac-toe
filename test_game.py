from random import choice 
from game import TicTacToe

game = TicTacToe()
while not game.is_done():
    # test checking on straights
    game.make_move(1)
    game.make_move(6)
    game.make_move(0)
    game.make_move(7)
    game.make_move(2)

print(f"winner is {"x" if game.check_winner() == 1 else "o"}")
game.reset()


while not game.is_done():
    # test checking on diagonals
    game.make_move(1)
    game.make_move(2)
    game.make_move(0)
    game.make_move(4)
    game.make_move(3)
    game.make_move(6)

print(f"winner is {"x" if game.check_winner() == 1 else "o"}")
game.reset()

while not game.is_done():
    # test checking on draw
    game.make_move(0)
    game.make_move(1)
    game.make_move(2)
    game.make_move(3)
    game.make_move(4)
    game.make_move(6)
    game.make_move(5)
    game.make_move(8)
    game.make_move(7)
    
print(f"winner is {"draw" if game.check_winner() == 0 else "broken"}")
game.reset()

game.make_move(0)
try:
    game.make_move(0)
    assert False, "illegal move was accepted!"
except ValueError:
    print("illegal move rejected")
    
print("illegal move check works too")
game.reset()

for _ in range(1000):
    game.reset()
    while not game.is_done():
        move = choice(game.get_valid_moves())
        game.make_move(move)
    assert game.check_winner() in (-1, 0, 1), f"game {_} broken: {game.check_winner()}"