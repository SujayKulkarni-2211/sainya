# SAINYA — सैन्य
### Where warriors are made of weights and biases

---

## What is this?

A war strategy game where every decision you make as a military commander secretly configures a real neural network. At the end, your model battles everyone else's on a real dataset. The winner had the best deep learning intuition — they just didn't know it.

**The reveal is the teaching moment.**

---

## Two ways to run

### Option A: Frontend Only (Demo Mode — works instantly, no setup)
Just open `sainya.html` in any browser. No server needed. The ML results are simulated locally based on your choices. Perfect for a demo where you want to show the concept without running a server.

### Option B: Full Version (Real ML training)

**Requirements:**
```
pip install fastapi uvicorn scikit-learn numpy
```

**Run backend:**
```
python sainya_backend.py
```

**Then open `sainya.html` in browser** — it will automatically connect to `localhost:8000`.

---

## The War → Deep Learning Mapping

| War Decision | DL Concept | Values |
|---|---|---|
| Attack Formation | Batch Size | Tight=32, Standard=64, Spread=128 |
| Advance Speed | Learning Rate | Slow=0.001, Medium=0.01, Fast=0.1 |
| Commander | Optimizer | Arjuna=SGD, Chanakya=Adam, Bhima=RMSprop |
| Command Layers | Network Depth | 1/2/3 hidden layers |
| Resource Discipline | L2 Regularization | Strict=0.01, Moderate=0.001, Loose=0.0001 |
| War Drills | Training Epochs | 50/150/300 |
| Retreat Strategy | Early Stopping | On/Off |

**The battlefield = Breast Cancer Wisconsin dataset (sklearn built-in)**
- 569 samples, 30 features, binary classification
- Trains in 1-3 seconds per model
- Accuracy range: ~72% to ~98% depending on choices

---

## Multiplayer Setup (for 20-60 people)

1. Run backend on your laptop
2. Use ngrok to expose it: `ngrok http 8000`
3. Change the API variable in sainya.html to your ngrok URL
4. Share the HTML file or host it anywhere
5. Create a room → share 6-letter code → everyone joins

---

## The 5-minute Demo Flow

1. Everyone joins with room code (1 min)
2. Everyone makes war decisions (2 min)
3. Armies deploy — real training happens (30 sec)
4. Leaderboard appears — reactions (1 min)
5. Hit "Reveal" — every decision explained in DL terms (1 min)
6. You teach the math (5 min)

---

## Deploy free on Railway / Render

Backend: Push to GitHub → deploy on Render.com (free tier)
Frontend: Host HTML on GitHub Pages / Netlify (free)

---

## The Math Behind Each Decision

**Batch Size:** `w ← w - η · ∇L(w; x_batch)`

**Learning Rate:** `θ_{t+1} = θ_t - η · ∇J(θ)`

**Adam Optimizer:** `θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)`

**L2 Regularization:** `L_total = L_CE + α · ||w||²`

**Backpropagation:** `∂L/∂W¹ = ∂L/∂h³ · ∂h³/∂h² · ∂h²/∂h¹ · ∂h¹/∂W¹`

---

Built for AIML juniors who deserve to discover, not to be taught.
