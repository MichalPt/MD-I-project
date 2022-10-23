# import dash
# from jupyter_dash import JupyterDash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.express as px
import pandas as pd
import numpy as np
# import plotly.graph_objects as go


### working ### - for pandas - multiindex

def format_to_pandas(data):
    output = None
    for config in data:
        col_labels = list()
        col_values = list()
        
        for i,n in enumerate(config):
            if type(n) == tuple:
                a, b = n
                col_labels.append(a)
                col_values.append(b)
            else:
                data = config[i:]
                break
        indices = pd.MultiIndex.from_frame(pd.DataFrame([col_values]), names=col_labels)
        pd_config = pd.DataFrame(data=[data], index=indices)
        
        if output is None:
            output = pd_config
        else:
            output = output.append(pd_config)
    return output




# plot interactive plot with plotly + dash + jupyter-lab:

# def plot_with_plotly(input_data):
    
#     if type(input_data) == str and ".csv" in input_data:
#         data = load_csv(input_data)
        
#     else:
#         data = format_to_pandas(input_data)

#     external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#     app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

#     server = app.server

#     option_setters = [
#         html.Div([html.Label(col_name+" : "),
#                   dcc.Checklist(
#                         id=col_name,
#                         options=[{"label": x, "value": x} 
#                                  for x in items],
#                         value=items,
#                         labelStyle={'display': 'inline-block', 'margin-right':'10px'},
#                         style={'margin-bottom':'8px'},
#                     )
#                  ],
#                  style={'display': 'inline-block', 'margin-right':'20px'}
#         ) for items, col_name in zip(data.index.levels, data.index.names)]

#     app.layout = html.Div([
#         html.Div(
#             html.Label("Settings - available data:"),
#             style={'margin-bottom':'10px', 'font-size':"18px", 'font-weight':'bold'}
#         ),
#         html.Div(
#             option_setters, 
#             style={'background-color':'#e6f2b1', 'border-bottom':'2px solid black', 'border-top':'2px solid black'}
#         ),
#         dcc.Graph(id="graph"),
#         ],
#         style={'text-align': 'center', 'width':'80%', 'margin-left':'auto', 'margin-right':'auto'}
#     )


#     @app.callback(
#         Output("graph", "figure"), 
#         [Input(col_name, "value") for col_name in data.index.names])
#     def update_chart(*args):
#         #print("ARGS:   ...   ", args)

#         largs = list(args)

#         # one of the parameters has to be a list ... for unknown reasons ...
#         if type(largs[0]) != list:
#             a = list()
#             a.append(largs[0])
#             largs[0] = a

#         sele = data.loc[tuple(largs), :]
#         labels = [sele.index.get_level_values(i).to_list() for i in sele.index.names]
#         new_order = np.argsort([len(np.unique(level)) for level in labels])
#         sele.index = sele.index.reorder_levels(new_order)

#         fig = go.Figure()

#         for config in sele.iterrows():
#             index, series = config
#             x_labels = np.repeat(np.array(index).reshape(len(index),1), len(series), axis=1)
#             fig.add_trace(go.Box(x= x_labels, y= series))

#         #fig.show()
#         fig.update_layout(title_text="Boxplot of MDS runtimes with selected parameters")
#         fig.update_layout(
#             autosize=False,
#             width=800,
#             height=500,
#             margin=dict(
#                 l=60,
#                 r=60,
#                 b=100,
#                 t=100,
#                 pad=4
#             ),
#         )

#         fig.update_yaxes(title_text='Time [s]')
#         fig.update_xaxes(title_text='MDS Settings')
#         fig.update_yaxes(rangemode='tozero')

#         return fig


#     app.run_server(mode="jupyterlab")
#     #app.run_server(mode="inline")


    
    
# def load_csv(filepath, sep=','):
#     with open(filepath, 'r') as file:
#         line = next(file)
#         cut = line.split(sep)

#     for i, x in enumerate(cut):
#         if len(x) == 1:
#             if int(x) == 0:
#                 index = i
#                 break
#             else:
#                 print("ERROR in csv header analysis!!!")

#     data = pd.read_csv(filepath, sep=sep, index_col=list(range(0,index)) )
#     return data

        
