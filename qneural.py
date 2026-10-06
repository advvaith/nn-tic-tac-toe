import torch
import torch.nn as nn
from game import TicTacToe
from main import NeuralNetwork
import random
import collections

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
buffer = collections.deque(maxlen=10000)
gamma = 0.9

def pick_move(game, model, eps=0.1):
    with torch.no_grad():
        scores = model(torch.tensor([cell * game.player for cell in game.board], dtype=torch.float32))
    valid = game.get_valid_moves()
    if random.random() < eps:
        return random.choice(valid), scores
    return max(valid, key=lambda c: scores[c].item()), scores

def play_episode(model, eps=0.1):
    game = TicTacToe()
    history = []
    
    while not game.is_done():
        board_before = game.board.copy()
        player = game.player
        move, _ = pick_move(game, model, eps)
        game.make_move(move)
        history.append((board_before, player, move))

    return history, game.check_winner(), game.board

def add_rewards(history, winner, final_board):
    samples = []
    for i, (board, player, move) in enumerate(history):
        board = [cell * player for cell in board]
        if i == len(history) - 1:
            done = True
            next_board = final_board
        else:
            done = False
            next_board = [cell * (-player) for cell in history[i + 1][0]]
        
        reward = 0.0
        if winner != 0:
            reward = 1.0 if player == winner else -1.0
        samples.append((board, move, reward, next_board, done))
    return samples

def prepare_batch(samples):
    # boards = [s[0] for s in samples]
    # moves = [s[1] for s in samples]
    # rewards = [s[2] for s in samples]
    
    boards, moves, rewards, final_board, done = zip(*samples) # this line does the same as all the above code; just that the rewards arent scaled yet
    
    board_tensor = torch.tensor(boards, dtype=torch.float32)
    move_tensor = torch.tensor(moves, dtype=torch.long)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    final_board_tensor = torch.tensor(final_board, dtype=torch.float32)
    done_tensor = torch.tensor(done, dtype=torch.bool)
    
    return (board_tensor, move_tensor, reward_tensor, final_board_tensor, done_tensor)
    
def train_step(samples):
    tensors = prepare_batch(samples)
    boards, moves, rewards = tensors[0], tensors[1], tensors[2]
    targets = compute_target(*prepare_batch(samples))
    scores = model(boards)
    N = boards.shape[0]
    picked = scores[torch.arange(N), moves]
    loss = criterion(picked, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
    
def evaluate(model, n_games=100):
    x_wins, draws, o_wins = 0, 0, 0
    
    for _ in range(n_games):
        game = TicTacToe()
        
        while not game.is_done():
            move, _ = pick_move(game, model, 0)
            if game.player == 1:
                game.make_move(move)
            else:
                game.make_move(random.choice(game.get_valid_moves()))
            
        winner = game.check_winner()
        if winner == 1:
            x_wins += 1
        elif winner == -1:
            o_wins += 1
        else:
            draws += 1
            
    print(f"x wins: {x_wins}, o wins: {o_wins}, draws: {draws}")
    
def train_round(n_episode=20):  
    for _ in range(n_episode):
        history, winner, final_board = play_episode(model)
        remember(add_rewards(history, winner, final_board))
    
    loss = []
    for _ in range(5):
        loss.append(train_step(sample_batch()))
    print(sum(loss)/5)

def remember(samples):
    buffer.extend(samples)
    
def sample_batch(size=64):
    return random.sample(buffer, size)

def compute_target(boards, moves, rewards, next_boards, dones):
    with torch.no_grad():
        next_q = model(next_boards)
        best_next = next_q.max(dim=1).values
        targets = rewards - gamma * best_next * (~dones).float()
        return targets
    
for round_num in range(200):
    train_round()
    if round_num % 20 == 0:
        print(f"round {round_num}")
        evaluate(model)
        
torch.save(model.state_dict(), "checkpoint_g0.9_lr0.01.pt")
model.load_state_dict(torch.load("checkpoint_g0.9_lr0.01.pt"))
