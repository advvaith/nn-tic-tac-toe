from fastapi import FastAPI
import torch
import json
from pathlib import Path
from pydantic import BaseModel
from main import NeuralNetwork

app = FastAPI()

import os
WALL_FILE = Path(os.environ.get("WALL_PATH", Path(__file__).parent / "wall.json"))

def load_wall() -> dict:
    if WALL_FILE.exists():
        return json.loads(WALL_FILE.read_text())
    return {}

class WallEntry(BaseModel):
    name: str

@app.get('/wall')
def get_wall():
    return load_wall()

@app.post('/wall')
def add_wall(entry: WallEntry):
    wall = load_wall()
    wall[entry.name] = wall.get(entry.name, 0) + 1
    WALL_FILE.write_text(json.dumps(wall, indent=2))
    return wall

# two seat specialists: xseat answers as X (second mover), oseat opens as O (first mover)
models = {
    "x": NeuralNetwork(),
    "o": NeuralNetwork(),
}
models["x"].load_state_dict(torch.load("best.pt", map_location='cpu'))
models["o"].load_state_dict(torch.load("best_o.pt", map_location='cpu'))
for m in models.values():
    m.eval()

class MoveRequest(BaseModel):
    board: list[int]
    valid_moves: list[int]
    seat: str = "x"   

from fastapi.responses import FileResponse
from minmax import choose_move

@app.get('/')
def home():
    return FileResponse("frontend/index.html")


@app.get('/explain')
def explain():
    return FileResponse("frontend/explain.html")


@app.post('/move')
def move(req: MoveRequest):
    if req.seat == "perfect":
        # board arrives in mover's perspective (mover stones = +1), so the minimax mover is player 1
        return {"move": choose_move(req.board, 1)}
    model = models[req.seat]
    scores = model(torch.tensor(req.board, dtype=torch.float32))
    best = max(req.valid_moves, key= lambda c: scores[c].item())
    return {"scores": scores.tolist(), "move": best}