# pitch_predictor
## Deep Learning Pitch Prediction Model 

![](https://github.com/nchapman95/pitch_predictor/blob/master/southparkpitch%20(1).gif)

Did some work while studying in my Deep Learning class. We were learning about Artificial Neural Networks,
and I was looking up some practical applications that I could easily implement. 

I came across a project that attempted to predict pitches from statcast data based on various elements of a baseball game. 
These elements/features/variables include how many runners there are on base, the last pitch thrown, the handedness of the
batter and the pitcher etc, and attempted to predict the pitch type for a single pitcher. The writer bucketed the pitches
into general pitch types: Breaking, Fast, Changeup. 

I thought this particular approach was intriguing, and solved some of the problems around identifying pitches. Often, there
can be misclassifications between types of breaking ball pitches for example. 

I had two takeaways from reading this writer's analysis. 

  1. There was only one pitcher in the study. Why not more than that?
  
  2. I wish the writer had reported his confusion matrix, or accuracy for each class of pitch,
     because I suspect his accuracy was due to the model predicting fastball
     (the preferred pitch of the pitcher in question) for every pitch, thus getting a 60% accuracy. 

Anyway, here is my attempt at building a similar model, attempting to predict a breaking ball, fastball, or changeup during any at-bat in a
Major League Baseball Game.

---

## MLB Game Predictor (`game_predictor` branch)

A full-stack web app that shows live MLB moneyline odds from major sportsbooks alongside ML model predictions for today's winners.

### Stack
- **Backend**: FastAPI (Python) — serves odds + predictions via REST
- **Frontend**: React + Vite — real-time dashboard, auto-refreshes every 60s
- **ML Model**: GradientBoosting classifier trained on pybaseball team stats (batting avg, OBP, SLG, ERA, WHIP, etc.)
- **Odds Data**: [The Odds API](https://the-odds-api.com) — DraftKings, FanDuel, BetMGM, Caesars

### Setup

**1. Get a free Odds API key**

Sign up at https://the-odds-api.com (free tier = 500 requests/month).

**2. Configure your key**

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set ODDS_API_KEY=your_key_here
```

**3. Install backend dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**4. Train the ML model** (downloads historical data via pybaseball — takes ~10 min first run)

```bash
cd backend
python -m model.train --years 2019 2020 2021 2022 2023 2024
```

**5. Start the backend**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**6. Install and start the frontend**

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/games` | Today's games with live odds + ML predictions |
| `GET /api/odds/raw` | Raw response from The Odds API |
| `GET /api/health` | Backend health + model status |

### ML Model Details

- **Target**: Home team win (binary classification)
- **Features**: Home/away team batting (AVG, OBP, SLG, BB%, K%) and pitching (ERA, WHIP, K/9, BB/9, HR/9, FIP) for the current season
- **Algorithm**: Gradient Boosting Classifier (sklearn)
- **Expected accuracy**: ~57–60% (MLB is hard to predict — anything above 55% is meaningful)
- **Fallback**: If model isn't trained yet, uses historical home-field advantage (~54%)

### Project Structure

```
pitch_predictor/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── odds_client.py       # The Odds API wrapper
│   ├── requirements.txt
│   └── model/
│       ├── features.py      # Feature engineering (pybaseball)
│       ├── train.py         # Training script
│       ├── predictor.py     # Inference
│       └── artifacts/       # Saved model (generated after training)
└── frontend/
    └── src/
        ├── App.jsx           # Main app + polling logic
        └── components/
            └── GameCard.jsx  # Per-game odds + prediction display
```
