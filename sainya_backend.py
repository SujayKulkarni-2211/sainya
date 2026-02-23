from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import json
import asyncio
import uuid
from typing import Dict, List, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load dataset once
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Game rooms
rooms: Dict[str, dict] = {}
connections: Dict[str, List[WebSocket]] = {}

class WarDecisions(BaseModel):
    room_id: str
    player_id: str
    player_name: str
    attack_formation: int      # 1=tight(32), 2=standard(64), 3=spread(128) -> batch_size
    advance_speed: str         # slow/medium/fast -> learning_rate 0.001/0.01/0.1
    commander: str             # arjuna/chanakya/bhima -> sgd/adam/rmsprop (we use adam with params)
    command_layers: int        # 1/2/3 -> hidden_layer_sizes
    troop_rotation: float      # 0.0/0.2/0.4 -> dropout (approximated via noise)
    supply_discipline: str     # strict/moderate/loose -> alpha L2 0.01/0.001/0.0001
    war_drills: int            # 50/150/300 -> max_iter
    retreat_strategy: bool     # True/False -> early_stopping

def decisions_to_hyperparams(d: WarDecisions):
    batch_map = {1: 32, 2: 64, 3: 128}
    lr_map = {"slow": 0.001, "medium": 0.01, "fast": 0.1}
    layers_map = {1: (64,), 2: (128, 64), 3: (256, 128, 64)}
    alpha_map = {"strict": 0.01, "moderate": 0.001, "loose": 0.0001}
    
    return {
        "hidden_layer_sizes": layers_map.get(d.command_layers, (128, 64)),
        "learning_rate_init": lr_map.get(d.advance_speed, 0.01),
        "alpha": alpha_map.get(d.supply_discipline, 0.001),
        "max_iter": d.war_drills,
        "early_stopping": d.retreat_strategy,
        "validation_fraction": 0.1 if d.retreat_strategy else 0.0,
        "batch_size": batch_map.get(d.attack_formation, 64),
        "solver": "adam",
        "random_state": 42,
        "n_iter_no_change": 10
    }

def train_model(decisions: WarDecisions) -> dict:
    params = decisions_to_hyperparams(decisions)
    clf = MLPClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    iterations = clf.n_iter_
    return {
        "accuracy": round(acc * 100, 2),
        "iterations": iterations,
        "params": params
    }

def get_dl_explanation(decisions: WarDecisions, result: dict) -> dict:
    lr_val = {"slow": 0.001, "medium": 0.01, "fast": 0.1}[decisions.advance_speed]
    batch_val = [32, 64, 128][decisions.attack_formation - 1]
    formation_desc = "tight formation (32)" if decisions.attack_formation == 1 else "standard formation (64)" if decisions.attack_formation == 2 else "spread formation (128)"
    batch_desc = "Small batches = noisy gradient updates, better exploration, escape local minima." if decisions.attack_formation == 1 else "Medium batches = balanced stability and exploration." if decisions.attack_formation == 2 else "Large batches = stable but may converge to sharp minima."
    speed_desc = "slow, precise speed" if decisions.advance_speed == "slow" else "standard marching speed" if decisions.advance_speed == "medium" else "aggressive full speed"
    lr_desc = "Too slow = takes forever, too fast = overshoots optimal weights." if decisions.advance_speed == "fast" else "Good choice for stable convergence." if decisions.advance_speed == "medium" else "Precise but slow convergence."
    layers_desc = "Shallow = simple patterns only." if decisions.command_layers == 1 else "Deep = complex feature extraction, risk of vanishing gradients." if decisions.command_layers == 3 else "Balanced depth."
    discipline_label = "Strict" if decisions.supply_discipline == "strict" else "Moderate" if decisions.supply_discipline == "moderate" else "Loose"
    discipline_desc = "Strong regularization = simpler model, less overfitting." if decisions.supply_discipline == "strict" else "Balanced regularization." if decisions.supply_discipline == "moderate" else "Weak regularization = risk of overfitting."
    drills_desc = "Undertrained = underfitting." if decisions.war_drills == 50 else "Well trained." if decisions.war_drills == 150 else "Risk of overfitting if early stopping not used."
    retreat_war = "Retreat strategy ACTIVE — army pulled back when weakening" if decisions.retreat_strategy else "No retreat — army fought until the end"
    retreat_dl = "Monitors validation loss. Stops training when model stops improving. Prevents overfitting." if decisions.retreat_strategy else "Trained for full epochs. Risk of overfitting on training data."
    explanations = {
        "attack_formation": {
            "war": f"You deployed warriors in {formation_desc}",
            "dl": f"Batch Size = {batch_val}. {batch_desc}",
            "math": "w = w - η · ∇L(w; x_batch)"
        },
        "advance_speed": {
            "war": f"Your army advanced at {speed_desc}",
            "dl": f"Learning Rate η = {lr_val}. {lr_desc}",
            "math": "η controls step size: θ_{t+1} = θ_t - η · ∇J(θ)"
        },
        "command_layers": {
            "war": f"Your army had {decisions.command_layers} {'layer' if decisions.command_layers == 1 else 'layers'} of command hierarchy",
            "dl": f"Hidden Layers = {decisions.command_layers}. Architecture: {result['params']['hidden_layer_sizes']}. {layers_desc}",
            "math": "h^(l) = σ(W^(l) · h^(l-1) + b^(l))"
        },
        "supply_discipline": {
            "war": f"{discipline_label} resource discipline across your kingdom",
            "dl": f"L2 Regularization α = {result['params']['alpha']}. Adds penalty α||w||² to loss. {discipline_desc}",
            "math": "L_total = L_CE + α·||w||²"
        },
        "war_drills": {
            "war": f"Your army trained for {decisions.war_drills} drills before battle",
            "dl": f"Max Epochs = {decisions.war_drills}. Model actually trained for {result['iterations']} iterations. {drills_desc}",
            "math": "One epoch = full pass through training data"
        },
        "retreat_strategy": {
            "war": retreat_war,
            "dl": f"Early Stopping = {'ON' if decisions.retreat_strategy else 'OFF'}. {retreat_dl}",
            "math": "Stop if val_loss doesn't improve for n_iter_no_change steps"
        }
    }
    return explanations

class JoinRoom(BaseModel):
    room_id: str
    player_id: str
    player_name: str

@app.post("/create_room")
async def create_room():
    room_id = str(uuid.uuid4())[:6].upper()
    rooms[room_id] = {"players": {}, "status": "waiting", "results": {}}
    connections[room_id] = []
    return {"room_id": room_id}

@app.post("/join_room")
async def join_room(data: JoinRoom):
    room_id = data.room_id.upper()
    if room_id not in rooms:
        return {"ok": False, "error": "Room not found. Check the code and try again."}
    if rooms[room_id]["status"] == "started":
        return {"ok": False, "error": "This room's campaign has already begun."}
    # Register player in room
    rooms[room_id]["players"][data.player_id] = {"name": data.player_name, "accuracy": None}
    # Broadcast updated player list to everyone in the room
    player_list = [{"pid": pid, "name": p["name"]} for pid, p in rooms[room_id]["players"].items()]
    await broadcast_to_room(room_id, {"type": "player_list", "players": player_list})
    return {"ok": True, "room_id": room_id, "players": player_list}

async def broadcast_to_room(room_id: str, msg: dict):
    if room_id not in connections:
        return
    dead = []
    for ws in connections[room_id]:
        try:
            await ws.send_json(msg)
        except:
            dead.append(ws)
    for ws in dead:
        connections[room_id].remove(ws)

@app.post("/battle")
async def battle(decisions: WarDecisions):
    result = train_model(decisions)
    explanation = get_dl_explanation(decisions, result)

    room_id = decisions.room_id
    if room_id not in rooms:
        rooms[room_id] = {"players": {}, "status": "battling", "results": {}}

    # Store result
    rooms[room_id]["results"][decisions.player_id] = {
        "name": decisions.player_name,
        "accuracy": result["accuracy"],
        "iterations": result["iterations"],
    }
    # Also update player entry
    if decisions.player_id in rooms[room_id]["players"]:
        rooms[room_id]["players"][decisions.player_id]["accuracy"] = result["accuracy"]

    # Broadcast result to all in room
    await broadcast_to_room(room_id, {
        "type": "player_result",
        "player": decisions.player_name,
        "player_id": decisions.player_id,
        "accuracy": result["accuracy"]
    })

    # Build current leaderboard
    all_results = [{"name": v["name"], "accuracy": v["accuracy"]}
                   for v in rooms[room_id]["results"].values()]
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)

    return {
        "accuracy": result["accuracy"],
        "iterations": result["iterations"],
        "explanation": explanation,
        "room_results": all_results
    }

@app.get("/room/{room_id}")
async def get_room(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        return {"error": "Room not found"}
    room = rooms[room_id]
    player_list = [{"pid": pid, "name": p["name"], "accuracy": p.get("accuracy")}
                   for pid, p in room["players"].items()]
    results = [{"name": v["name"], "accuracy": v["accuracy"]}
               for v in room.get("results", {}).values()]
    results.sort(key=lambda x: x["accuracy"] if x["accuracy"] is not None else 0, reverse=True)
    total_players = len(room["players"])
    finished_players = len(room.get("results", {}))
    return {
        "players": player_list,
        "results": results,
        "status": room["status"],
        "total_players": total_players,
        "finished_players": finished_players
    }

@app.post("/start_room/{room_id}")
async def start_room(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        return {"error": "Room not found"}
    rooms[room_id]["status"] = "started"
    await broadcast_to_room(room_id, {"type": "start_game"})
    return {"ok": True}

@app.websocket("/ws/{room_id}/{player_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, player_id: str):
    await websocket.accept()
    room_id = room_id.upper()
    if room_id not in connections:
        connections[room_id] = []
    connections[room_id].append(websocket)

    # Register player in room if not already there
    if room_id in rooms and player_id not in rooms[room_id]["players"]:
        # Will be registered via /join_room, but just in case
        pass

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            # Broadcast to all others in room
            for ws in connections[room_id]:
                if ws != websocket:
                    try:
                        await ws.send_text(data)
                    except:
                        pass
    except WebSocketDisconnect:
        if websocket in connections.get(room_id, []):
            connections[room_id].remove(websocket)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def serve_frontend():
    base = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base, "sainya.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
