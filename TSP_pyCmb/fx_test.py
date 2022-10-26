import time

import plotly.graph_objs as go
import plotly.io as pio
from sklearn import datasets
import pandas as pd
import ipywidgets

pio.renderers.default = 'browser'
f = go.FigureWidget()
f.add_scatter(y=[2, 1, 4, 3])
f.add_bar(y=[1, 4, 3, 2])
f.layout.title = 'Hello FigureWidget'
f.show()
# update scatter data
time.sleep(2)
with f.batch_update():
    scatter = f.data[0]
    scatter.y = [3, 1, 4, 3]
    # update bar data
    bar = f.data[1]
    bar.y = [5, 3, 2, 8]
    f.layout.title = 'This is a new title'


iris_data = datasets.load_iris()
feature_names = [name.replace(' (cm)', '').replace(' ', '_') for name in iris_data.feature_names]
iris_df = pd.DataFrame(iris_data.data, columns=feature_names)
iris_class = iris_data.target + 1
fig = go.FigureWidget()

fig.add_scatter(x=iris_df.sepal_length, y=iris_df.petal_width)
fig.show()
scatter = fig.data[0]
time.sleep(2)
scatter.mode = 'markers'
scatter.marker.size = 8
scatter.marker.color = iris_class
scatter.marker.cmin = 0.5
scatter.marker.cmax = 3.5
scatter.marker.colorscale = [[0, 'red'], [0.33, 'red'],
                             [0.33, 'green'], [0.67, 'green'],
                             [0.67, 'blue'], [1.0, 'blue']]