import json
from benderclient import Bender
from random import *
import plotly.plotly as py
import plotly.graph_objs as go
from pandas import DataFrame
import numpy as np
​
# Interesting finction to try : np.exp(-(x-1)**2-y**2)-10*(x**3+y**4-x/5)*np.exp(-(x**2+y**2))
​
bender = Bender()
map_size = 100
nomansland = [[0 for i in range(map_size)] for j in range(map_size)]
bombs = 800
radius = 10
sugestions = []
n_suggestions = 200
​


def init_bender():
  bender.create_experiment(
      name='NoMansLand',
      description='',
      metrics=[{"metric_name": "altitude", "type": "loss"}],
      dataset='my'
  )
  bender.create_algo(
      name='Benchmark',
      hyper_parameters=[
          {
              "name": 'x',
              "category": "uniform",
              "search_space": {
                  "low": 0,
                  "high": map_size,
                  "step": 1,
              },
          },
          {
              "name": 'y',
              "category": "uniform",
              "search_space": {
                  "low": 0,
                  "high": map_size,
                  "step": 1,
              },
          },
      ]
  )


​


def play_fortunate_son():
  for b in range(bombs):
    power = uniform(0, 1)
    x = randint(0, map_size)
    y = randint(0, map_size)
    for j in range(-radius, radius):
      for i in range(-radius, radius):
        try:
          ease = pow((pow(radius, 2) * 2 - (pow(j, 2) + pow(i, 2))) * power, 2)
          nomansland[y + j][x + i] = nomansland[y + j][x + i] - ease
        except IndexError:
          pass


​
play_fortunate_son()
init_bender()
​
for i in range(n_suggestions):
  s = bender.suggest(metric="altitude")
  sugestions.append(s)
  bender.create_trial(
      hyper_parameters=s,
      results={"altitude": float(nomansland[int(s["y"])][int(s["x"])])}
  )
with open("sugestions.json", "a") as myfile:
  myfile.write(json.dumps(sugestions, ensure_ascii=False))
​
with open("dataset.json", "a") as myfile:
  myfile.write(json.dumps(nomansland, ensure_ascii=False))
​
X = []
Y = []
Z = []
for i in range(n_suggestions):
  X.append(int(sugestions[i]["x"]))
  Y.append(int(sugestions[i]["y"]))
  Z.append(float(nomansland[int(sugestions[i]["y"])][int(sugestions[i]["x"])]))
trace2 = go.Scatter3d(
    x=X,
    y=Y,
    z=Z,
    mode='markers',
    marker=dict(
        color='rgb(255,255,0)',
        size=2,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=1
    )
)
​
data = [
    go.Surface(
        z=nomansland
    ),
    trace2
]
layout = go.Layout(
    title='No Mans Land',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='elevations-3d-surface')
