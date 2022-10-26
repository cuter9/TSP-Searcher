############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: pyCombinatorial - Graphs

# GitHub Repository: <https://github.com/Valdecy>

############################################################################

# Required Libraries
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go


############################################################################
# Function: Plot Search Evolution
def plot_evolution(evolution_profile, best_solution, Optimal_cost, best_search_idx, search_method_name):
    if not evolution_profile:
        return
    pio.renderers.default = "browser"  # opens figures in a tab of the default web browser
    # Default renderers persist for the duration of a single session,
    # but they do not persist across sessions.
    # If you are working in an IPython kernel,
    # this means that default renderers will persist for the life of the kernel,
    # but they will not persist across kernel restarts.
    # plotly.tools.set_credentials_file(username='cuter490703', api_key='cjm8RyxPeAIgJQx3DwMu')
    # x_line = np.linspace(0, generation, generation)
    # Create traces
    fig_title = 'Evolution Profile with ' + search_method_name
    # title = fig_title
    # fig_title = fig_data['fig_title']
    # y_scale = fig_data['y_scale']
    # file_name = fig_data['file_name']
    y_label = 'Distance'
    # y_label = fig_data['y_label']

    layout = go.Layout(title=fig_title,
                       legend=dict(font=dict(size=12)),
                       xaxis=dict(title='generation',
                                  showline=True,
                                  showgrid=False,
                                  showticklabels=True,
                                  linecolor='rgb(204, 204, 204)',
                                  ticks='outside',
                                  tickcolor='rgb(204, 204, 204)',
                                  ticklen=5,
                                  tickfont=dict(family='Arial',
                                                size=12,
                                                color='rgb(82, 82, 82)',
                                                ),
                                  ),
                       yaxis=dict(title=y_label,
                                  # type='log',
                                  # exponentformat='E',
                                  showgrid=True,
                                  zeroline=True,
                                  showline=True,
                                  showticklabels=True,
                                  ticks='outside'
                                  ),
                       autosize=True,
                       margin=dict(autoexpand=True,
                                   l=50,
                                   r=50,
                                   b=50,
                                   t=50,
                                   ),
                       showlegend=True)

    fig = go.Figure(layout=layout)
    for i in range(0, len(evolution_profile)):
        if i == best_search_idx:
            line_color = 'firebrick'
            dash_line = 'solid'
            w_line = 2
            y_label_line = 'Best Distance'
        else:
            line_color = 'lightblue'
            dash_line = 'dashdot'
            w_line = 1
            y_label_line = 'Distance'
        fig.add_trace(go.Scatter(x=evolution_profile[i][0][0],
                                 y=evolution_profile[i][0][1],
                                 mode='lines+markers',
                                 name=y_label_line,
                                 line=dict(color=line_color, width=w_line, dash=dash_line)
                                 )
                      )

    fig.add_annotation(text='Best Search : ' + '<br>' +
                            'Shortest Distance = ' + str(Optimal_cost) + ' km' + '<br>' +
                            'Searched Best Distance = ' + str(best_solution[best_search_idx][1]) + ' km' + '<br>' +
                            'Cost Gap = ' + str(best_solution[best_search_idx][2]) + ' %' + '<br>',
                       align='left',
                       showarrow=False,
                       xref='paper',
                       yref='paper',
                       x=0.95,
                       y=0.9,
                       bordercolor='blue',
                       borderwidth=1
                       )
    # fig = go.FigureWidget(fig)
    file_name = 'Results/' + fig_title + '.html'
    fig.write_html(file_name)
    # pio.write_image(fig, file_name)
    fig.show()
    # fig.update_traces(data = profile, layout=layout, overwrite = True)
    # os.chdir('F:\Courses\勤益科大\Artificial Intelligence\Lectures\Ch 7\Lecture Demo - 2019\Figures')
    # fig.show()  # different plot should have with different file name
    # os.chdir('F:\Courses\勤益科大\Artificial Intelligence\Lectures\Ch 7\Lecture Demo - 2019')


############################################################################

# Function: Build Coordinates
def build_coordinates(distance_matrix):
    a = distance_matrix[0, :].reshape(distance_matrix.shape[0], 1)
    b = distance_matrix[:, 0].reshape(1, distance_matrix.shape[0])
    m = (1 / 2) * (a ** 2 + b ** 2 - distance_matrix ** 2)
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1])) ** (1 / 2)
    coordinates = np.matmul(u, s ** (1 / 2))
    coordinates = coordinates.real[:, 0:2]
    return coordinates


############################################################################

# Function: Solution Plot 
def plot_tour(coordinates, best_solution=[], Optimal_cost=0, search_method_name='', view='browser', size=10):
    city_tour = best_solution[0]
    if coordinates.shape[0] == coordinates.shape[1]:
        coordinates = build_coordinates(coordinates)
    if view == 'browser':
        pio.renderers.default = 'browser'
    if len(city_tour) > 0:
        xy = np.zeros((len(city_tour), 2))
        for i in range(0, len(city_tour)):
            if i < len(city_tour):
                xy[i, 0] = coordinates[city_tour[i] - 1, 0]
                xy[i, 1] = coordinates[city_tour[i] - 1, 1]
            else:
                xy[i, 0] = coordinates[city_tour[0] - 1, 0]
                xy[i, 1] = coordinates[city_tour[0] - 1, 1]
    else:
        xy = np.zeros((coordinates.shape[0], 2))
        for i in range(0, coordinates.shape[0]):
            xy[i, 0] = coordinates[i, 0]
            xy[i, 1] = coordinates[i, 1]
    data = []
    Xe = []
    Ye = []
    ids = ['id: ' + str(i + 1) + '<br>' + 'x: ' + str(round(coordinates[i, 0], 2)) + '<br>' + 'y: ' + str(
        round(coordinates[i, 1], 2)) for i in range(0, coordinates.shape[0])]
    if len(city_tour) > 0:
        id0 = 'id: ' + str(city_tour[0]) + '<br>' + 'x: ' + str(round(xy[0, 0], 2)) + '<br>' + 'y: ' + str(
            round(xy[0, 1], 2))
    else:
        id0 = 'id: ' + str(1) + '<br>' + 'x: ' + str(round(xy[0, 0], 2)) + '<br>' + 'y: ' + str(round(xy[0, 1], 2))
    if len(city_tour) > 0:
        for i in range(0, xy.shape[0] - 1):
            Xe.append(xy[i, 0])
            Xe.append(xy[i + 1, 0])
            Xe.append(None)
            Ye.append(xy[i, 1])
            Ye.append(xy[i + 1, 1])
            Ye.append(None)
        e_trace = go.Scatter(x=Xe[2:],
                             y=Ye[2:],
                             mode='lines',
                             line=dict(color='rgba(0, 0, 0, 1)', width=0.50, dash='solid'),
                             hoverinfo='none',
                             name=''
                             )
        data.append(e_trace)
    n_trace = go.Scatter(x=coordinates[0:, -2],
                         y=coordinates[0:, -1],
                         opacity=1,
                         mode='markers+text',
                         marker=dict(symbol='circle-dot', size=size, color='rgba(46, 138, 199, 1)'),
                         hoverinfo='text',
                         hovertext=ids[0:],
                         name=''
                         )
    data.append(n_trace)
    m_trace = go.Scatter(x=xy[0:1, -2],
                         y=xy[0:1, -1],
                         opacity=1,
                         mode='markers+text',
                         marker=dict(symbol='square-dot', size=size, color='rgba(247, 138, 54, 1)'),
                         hoverinfo='text',
                         hovertext=id0,
                         name=''
                         )
    data.append(m_trace)
    fig_title = 'Searched Best Tour with ' + search_method_name
    layout = go.Layout(title=fig_title,
                       showlegend=False,
                       hovermode='closest',
                       plot_bgcolor='rgb(235, 235, 235)',
                       xaxis=dict(showgrid=True,
                                  zeroline=True,
                                  showticklabels=True,
                                  tickmode='array',
                                  ),
                       yaxis=dict(showgrid=True,
                                  zeroline=True,
                                  showticklabels=True,
                                  tickmode='array',
                                  ),
                       margin=dict(autoexpand=True,
                                   l=50,
                                   r=50,
                                   b=50,
                                   t=50,
                                   )

                       )
    fig = go.Figure(data=data, layout=layout)
    if len(city_tour) > 0:
        fig.add_annotation(
            x=Xe[1] * 1.00,  # to x
            y=Ye[1] * 1.00,  # to y
            ax=Xe[0] * 1.00,  # from x
            ay=Ye[0] * 1.00,  # from y
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='red',
            opacity=1
        )

    fig.update_traces(textfont_size=10, textfont_color='rgb(255, 255, 255)')
    fig.add_annotation(text='Best Tour : ' +
                            'Shortest Distance = ' + str(Optimal_cost) + ' km' + '<br>' +
                            'Searched Best Distance = ' + str(best_solution[1]) + ' km' + '<br>' +
                            'Cost Gap = ' + str(best_solution[2]) + ' %' + '<br>',
                       font=dict(size=12),
                       align='left',
                       xref='paper',
                       yref='paper',
                       showarrow=False,
                       x=0.95,
                       y=0.9,
                       bordercolor='blue',
                       borderwidth=1
                       )
    file_name = 'Results/' + fig_title + '.html'
    fig.write_html(file_name)
    # pio.write_image(fig, file_name)

    fig.show()
    return

############################################################################
