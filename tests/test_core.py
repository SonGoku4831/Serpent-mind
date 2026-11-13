import numpy as np
from fastapi.testclient import TestClient
from snake.game import Snake
from snake.agent import Agent

def test_food_and_collision():
    game=Snake()
    game.food=(6,5)
    assert game.step(0)==10
    assert game.score==1 and len(game.body)==4
    while not game.done:
        game.step(0)
    assert game.body[0][0]==9

def test_tail_cell_is_legal():
    game=Snake()
    game.body=[(3,3),(3,4),(2,4),(2,3)]
    game.direction=3
    game.food=(8,8)
    assert not game.danger(2)
    game.step(2)
    assert not game.done

def test_update_and_roundtrip(tmp_path):
    agent=Agent()
    agent.learn(1,2,10,2,True)
    assert agent.q[1,2]==1
    path=tmp_path/'q.npz'
    np.savez_compressed(path,q=agent.q)
    assert Agent.load(path).act(1)==2

def test_api(tmp_path,monkeypatch):
    path=tmp_path/'q.npz'
    np.savez_compressed(path,q=Agent().q)
    monkeypatch.setenv('MODEL_PATH',str(path))
    from snake.api import app
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        game=client.post('/games',json={'seed':1}).json()
        key=game['id']
        while not game['done']:
            response=client.post(f'/games/{key}/step')
            assert response.status_code==200
            game=response.json()
        assert client.post(f'/games/{key}/step').status_code==404
        metrics=client.get('/metrics').text
        assert 'snake_games_total' in metrics and 'snake_score_count' in metrics
