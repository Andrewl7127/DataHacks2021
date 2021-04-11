import pandas as pd
import base64

pd.options.mode.chained_assignment = None  # default='warn'

from dash.dependencies import Input, Output, State
import plotly.express as px
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pickle

v4 = pd.read_csv('prosperity.csv')
df = pd.read_csv('https://raw.githubusercontent.com/Andrewl7127/UCSD-DataHacks-2021/main/Data/merged.csv')
fig = px.line(v4, x='year', y='%_change_from_07', color='country', height=400,
              labels={'%_change_from_07': 'Prosperity % Change'},
              line_dash='country',
              title="Top 5 Countries By Prosperity Growth")

map1 = {"prosperity_score": 'Prosperity', "busi": 'Business Environment',
        "econ": 'Economic Quality', 'educ': 'Education', 'envi': 'Natural Environment', 'gove': 'Governance',
        "heal": "Health", 'pers': 'Personal Freedom', 'safe': 'Safety & Security', 'soci': 'Social Capital'}

map2 = {value: key for (key, value) in map1.items()}

image_filename = 'corr_matrix.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

filename = 'graph.pkl'
with open(filename, 'rb') as f:
    fig2 = pickle.load(f)

filename = 'animation.pkl'
with open(filename, 'rb') as f:
    fig3 = pickle.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

app.layout = html.Div(children=[
    html.H1('UCSD DataHacks 2021', style={'font-weight': 'bold', 'font-size': '350%', 'padding-left': '2px'}),
    html.H4('Created by Adhvaith Vijay, Andrew Liu, Shail Mirpuri, and Youngseo Do',
            style={'font-weight': 'bold', 'padding-left': '3px', 'font-size': '180%'}),
    html.Div([
        html.Label([dcc.Graph(figure=fig)]),
        html.Br(),
        html.Label([
            html.H4(children='Correlation Matrix Heatmap'),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ]),
        html.Br(),
        html.Label([dcc.Graph(figure=fig2)]),
        html.Br(),
        html.Label([dcc.Graph(figure=fig3)]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Label(["Select Score To Visualize",
                    dcc.Dropdown(id="metric",
                                 options=[{'label': k, 'value': k} for k in list(map1.values())],
                                 value="Prosperity",
                                 clearable=False,
                                 multi=False, style={'font-weight': 'normal'})], style={"width": "13%", 'font-weight': 'bold'}),
        html.Br(),
        html.Br(),
        html.Label(dbc.Button(id='button', n_clicks=0, children="Submit", color="primary")),
        html.Br(),
        html.Label([dcc.Graph(id='output', figure={})])], style={'text-align': 'center'}),
    html.Br()
])


@app.callback(
    Output(component_id='output', component_property='figure'),
    Input(component_id='button', component_property='n_clicks'),
    [State(component_id='metric', component_property='value')],
    prevent_initial_call=False
)
def viz_bucket(n, metric):
    converted = map2[metric]
    fig4 = px.choropleth(df, locations="isocode",
                         color=converted,  # lifeExp is a column of gapminder
                         hover_name="country",  # column to add to hover information
                         color_continuous_scale=px.colors.sequential.Plasma_r,
                         title=metric + ' Score (2014)')
    # fig4.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height = 500, width = 900)
    fig4.update_layout(height=600, width=950)
    return fig4


if __name__ == '__main__':
    app.run_server(debug=True)
