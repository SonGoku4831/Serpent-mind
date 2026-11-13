import os
import time
import numpy as np
import requests
import streamlit as st

API=os.getenv('API_URL','http://localhost:8000')
st.set_page_config(page_title='Snake AI',page_icon='🐍')
st.title('🐍 Watch AI play Snake')
st.write('Click Play to watch the trained reinforcement learning agent.')
seed=st.number_input('Game seed',min_value=0,max_value=2147483647,value=42)
speed=st.slider('Moves per second',1,30,10)
board=st.empty()
score=st.empty()

def render(game):
    pixels=np.full((game['size'],game['size'],3),22,dtype=np.uint8)
    for x,y in game['body']:
        pixels[y,x]=[36,180,90]
    x,y=game['body'][0]
    pixels[y,x]=[130,255,160]
    if game['food']:
        x,y=game['food']
        pixels[y,x]=[245,75,65]
    board.image(np.repeat(np.repeat(pixels,32,axis=0),32,axis=1),width=400)
    score.write(f"Score: {game['score']} | Moves: {game['steps']}")

if st.button('Play',type='primary'):
    key=None
    try:
        response=requests.post(f'{API}/games',json={'seed':int(seed)},timeout=10)
        response.raise_for_status()
        game=response.json()
        key=game['id']
        render(game)
        while not game['done']:
            time.sleep(1/speed)
            response=requests.post(f'{API}/games/{key}/step',timeout=10)
            response.raise_for_status()
            game=response.json()
            render(game)
        st.success(f"Game over — score {game['score']}")
    except requests.RequestException as exc:
        st.error(f'Cannot reach the game API: {exc}')
    finally:
        if key:
            try:
                requests.delete(f'{API}/games/{key}',timeout=5)
            except requests.RequestException:
                pass
