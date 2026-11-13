import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from snake.agent import Agent
from snake.game import Snake

GAMES=Counter('snake_games_total','Started games')
STEPS=Counter('snake_steps_total','Agent decisions')
SCORES=Histogram('snake_score','Finished game scores',buckets=(0,1,2,5,10,20,50,100))
LATENCY=Histogram('snake_inference_seconds','Agent decision latency')
ACTIVE=Gauge('snake_active_games','Active games')
sessions={}
lock=Lock()

@asynccontextmanager
async def lifespan(app):
    app.state.agent=Agent.load(os.getenv('MODEL_PATH','models/q.npz'))
    yield

app=FastAPI(title='Snake AI', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status':'ok'}

class NewGame(BaseModel):
    seed: int = Field(default=42, ge=0, le=2147483647)

def cleanup():
    now=time.monotonic()
    for key in list(sessions):
        if now-sessions[key][1] > 600:
            del sessions[key]
    ACTIVE.set(len(sessions))

@app.post('/games')
def create_game(request: NewGame):
    with lock:
        cleanup()
        if len(sessions)>=100:
            raise HTTPException(429,'Too many games; retry later')
        key=str(uuid.uuid4())
        game=Snake(request.seed)
        sessions[key]=(game,time.monotonic())
        GAMES.inc()
        ACTIVE.set(len(sessions))
        return {'id':key, **game.snapshot()}

@app.post('/games/{key}/step')
def step(key: str):
    with lock:
        cleanup()
        if key not in sessions:
            raise HTTPException(404,'Game ended or expired; start a new game')
        game,_=sessions[key]
        with LATENCY.time():
            action=app.state.agent.act(game.state())
        game.step(action)
        STEPS.inc()
        result=game.snapshot()
        if game.done:
            SCORES.observe(game.score)
            del sessions[key]
        else:
            sessions[key]=(game,time.monotonic())
        ACTIVE.set(len(sessions))
        return result

@app.delete('/games/{key}')
def remove(key: str):
    with lock:
        sessions.pop(key,None)
        ACTIVE.set(len(sessions))
    return {'status':'closed'}

@app.get('/metrics')
def metrics():
    with lock:
        cleanup()
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
