import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

from dash.dependencies import Input, Output
from collections import Counter
from simulate import *
from textwrap import dedent as d

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

folderStr = "./Data/"
destFolder = "./Precomputed Data/"

data_frame = pd.read_csv(folderStr + "RentalCars.csv")
data_frame_col = data_frame[["id", "brand", "model", "monthly_cost", "term", "upfront_cost"]]
data_frame_col.columns = ["id", "brand", "model", "monthly", "term", "upfront"]
utilityDf = readDF(folderStr + "UtilityScoresRC.csv")


survey_result = pd.DataFrame(utilityDf[['id', 'segment', 'current brand']].values,
                                      columns=['id', 'segment', 'current brand'])

per_prod_utility_matrix = np.genfromtxt(destFolder + "personProdMatch.csv", delimiter=",")

fusedMat = np.genfromtxt(destFolder + "utilityNumericValue.csv", delimiter=',')

baseLine_dataFrame = pd.read_csv(destFolder + "baselinePrediction.csv")

monthly_range = pd.Series([120, 200, 250, 400, 500, 750, 1000])
upfront_range = pd.Series([2000, 4000, 6000, 8000, 10000])
term_range = pd.Series([24, 36, 48])
worth_range = pd.Series([20000, 30000, 40000, 80000, 100000])
set_range = pd.Series([20, 40, 80, 180, 300])
charge_range = pd.Series([3, 10, 50, 100, 160])

num_attr = {'monthly_cost': monthly_range, 'upfront_cost': upfront_range, 'term': term_range,
                     'vehicle_worth': worth_range, 'range': set_range, 'charge': charge_range}

brands_category = pd.Series(['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW'])
energy_category = pd.Series(['Electric Vehicle', 'Plug-in Hybrid'])
type_category = pd.Series(['Sedan', 'SUV'])

cat_attr = {'brand': brands_category, 'energy': energy_category, 'vehicle_type': type_category}


colors = {'Audi': 'mediumblue', 'Chevrolet': 'gold', 'Jaguar': 'teal', 'Kia': 'deepskyblue',
              'Nissan': 'red', 'Tesla': 'purple', 'Toyota': 'orange', 'VW': 'deeppink'}
data_frame['colours'] = data_frame['brand'].map(colors)

### Dashbooard outline 
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div([
    html.H3(children="Rental Car Analysis", style={'textAlign': 'center'}),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='intermediate-value-2', style={'display': 'none'}),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Overview of Rental Cars', value='tab-1'),
        dcc.Tab(label='Brand Attraction & Attrition', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])


def PDtoDict(row):
    return eval("dict(" + data_frame_col.columns[0] + " = " + str(row.iat[0]) + ", "
                + data_frame_col.columns[1] + " = \'" + str(row.iat[1]) + "\', "
                + data_frame_col.columns[2] + " = \'" + str(row.iat[2]) + "\', "
                + data_frame_col.columns[3] + " = " + str(row.iat[3]) + ", "
                + data_frame_col.columns[4] + " = " + str(row.iat[4]) + ", "
                + data_frame_col.columns[5] + " = " + str(row.iat[5]) + ")")


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    
    if tab == 'tab-1':
        
        return html.Div([
            html.Div([
                dcc.Markdown(d("""
                **Rental Cars**\n

                """)),
                dash_table.DataTable(
                    id='products-input',
                    columns=(
                        [{'id': p, 'name': p} for p in data_frame_col.columns]
                    ),
                    data=[x for x in data_frame_col.apply(PDtoDict, axis=1)],
                    editable=True
                ),
                html.Div([
                    html.H6(children="---------------------------------", style={'textAlign': 'center'}),
                ], style={'padding': '1850px 0px 0px'}),
            ],
            style={'width': '25%', 'float': 'left', 'display': 'inline-block',
                    'borderRight': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '10px 20px 20px'
                    }
            ),
            
            html.H3(children="Market Share Predictions by Brand", style={'textAlign': 'center'}),
            html.Div([
                    dcc.Graph(
                            id='product-pie',
                            style={'height': 600}
                            )
                    ],style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
                    
            
            html.H3(children="Market Share Predictions by Product", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(
                    id='product-histogram',
                    style={
                        'height': 600
                    }
                )
            ],style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
            
            html.Div([dcc.Slider(id='Brand-selected', min=1, max=8, value=8,
                
                    marks={1: "Audi", 2: "Chevrolet", 3: "Jaguar", 4: "Kia", 5: "Nissan", 6: "Tesla", 7: "Toyota",
                                8: "VW"})
                ],style={'textAlign': "center", "margin": "30px", "padding": "10px", "margin-left": "auto",
                    "margin-right": "auto",'width': '70%', 'float': 'right', 'display': 'inline-block'}),
                 
        html.H3(children="Attraction towards the choosen product", style={'textAlign': 'center'}),
    
        html.Div([
                html.Div([
                    dcc.Graph(
                        id='heatmap',
                        style={
                            'height': 600,
                            'padding': '20px 20px 20px'
                        }
                    )
                ],
                style={'width': '70%', 'float': 'left', 'display': 'inline-block'}),
            ],
            style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
           
        ])
                
                
    elif tab == 'tab-2':
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H6(children="Brand Attraction", style={'textAlign': 'center'}),
                ], style={'padding': '0px 0px 0px'}),
                dcc.Dropdown(
                    id='brandBeauty',
                    options=[{'label': 'Major Brands', 'value': 'Major Brands'},
                             {'label': 'Minor Brands', 'value': 'Minor Brands'}],
                    value='Major Brands'
                ),
                html.Div([
                    html.H6(children="Brand Attrition", style={'textAlign': 'center'}),
                ], style={'padding': '800px 0px 0px'}),
                dcc.Dropdown(
                    id='brandAttrition',
                    options=[{'label': 'Audi', 'value': 'Audi'},
                             {'label': 'Chevrolet', 'value': 'Chevrolet'},
                             {'label': 'Jaguar', 'value': 'Jaguar'},
                             {'label': 'Kia', 'value': 'Kia'},
                             {'label': 'Nissan', 'value': 'Nissan'},
                             {'label': 'Tesla', 'value': 'Tesla'},
                             {'label': 'Toyota', 'value': 'Toyota'},
                             {'label': 'VW', 'value': 'VW'}],
                    value='Tesla'
                ),
                html.Div([
                    html.H6(children="---------------------------------", style={'textAlign': 'center'}),
                ], style={'padding': '700px 0px 0px'}),
            ],
            style={'width': '25%', 'float': 'left', 'display': 'inline-block',
                    'borderRight': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '0px 0px 0px'
                    }),
            
            html.H3(children="Strength of Attraction to chosen brand", style={'textAlign': 'center'}),
            html.Div([
                    dcc.Graph(
                            id='brandFig',
                            style={'height': 600}
                            )
                    ],style={'width': '70%', 'float': 'right', 'display': 'inline-block'}),
                    
            
            html.H3(children="Brand Scavangers", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(
                    id='brandWar',
                    style={
                        'height': 600
                    }
                )
            ],style={'width': '70%', 'float': 'right', 'display': 'inline-block'})
            
        ])




@app.callback(
    Output('intermediate-value', 'children'),
    [Input('products-input', 'data'),
     Input('products-input', 'columns')])
def calculateOffer(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    products_used_data_frame = data_frame.copy()
    copy_df = products_used_data_frame.copy()
    delta_index = np.argwhere((df['monthly'].values != copy_df['monthly_cost'].values) |
                           (df['term'].values != copy_df['term'].values) |
                           (df['upfront'].values != copy_df['upfront_cost'].values))
    delta_index = [item for sublist in delta_index for item in sublist]
    
    for i in delta_index:
        products_used_data_frame.at[i, 'monthly_cost'] = df.at[i, 'monthly']
        products_used_data_frame.at[i, 'term'] = df.at[i, 'term']
        products_used_data_frame.at[i, 'upfront_cost'] = df.at[i, 'upfront']
    
    aug_prod_data_frame = create_aug_prod_df(products_used_data_frame, cat_attr)
    utilityLease = add_utility(utilityDf, aug_prod_data_frame, num_attr, True, delta_index, fusedMat)
    pred_prod_val, pers_prod_match = pred_prod(aug_prod_data_frame, utilityDf, products_used_data_frame, utilityLease,
                                                   delta_index, per_prod_utility_matrix, baseLine_dataFrame)
    
    column_list = ['Prod' + str(x + 1) for x in range(pred_prod_val.shape[1] - 6)]
    pred_prod_val.columns = ['ID', 'Segment', 'BaseBrand', 'LiveBrand', 'BaseProduct', 'LiveProduct'] + column_list
    #print(pred_prod_val)
    return pred_prod_val.to_json(date_format='iso', orient='split')


@app.callback(Output('product-histogram', 'figure'),
              [Input('intermediate-value', 'children'),
               Input('Brand-selected','value')])
def display_histogram(JsonData,value):
    nP = data_frame.shape[0]
    pred_prod_val  = pd.read_json(JsonData, orient='split')
    products_used_data_frame = data_frame.copy()
    data_frame1 = pd.value_counts(pred_prod_val['LiveProduct']).to_frame().reset_index().sort_values('index')
    data_frame1.columns = ['prod ID', 'freq']
    data_frame1.index = data_frame1['prod ID']
    data_frame2 = pd.DataFrame({'prod ID': range(1, nP + 1, 1), 'freq': 0})
    data_frame2.index = data_frame2['prod ID']
    data_frame2.freq = data_frame1.freq
    data_frame2 = data_frame2.fillna(0)
    data_frame2.columns = ['prod ID', 'freq']
    
    car_brands =  Counter(products_used_data_frame['brand'])
    range_idx = sum(list(car_brands.values())[:value])
    modelX = ['e-tron', 'Bolt', 'I-Pace', 'Niro EV', 'Niro Plg', 'Optima', 'Leaf S', 'Leaf S+', 'Model 3', 'Model S',
            'Model X', 'Prius', 'e-Golf']
    track_down = go.Bar(x=modelX[:range_idx], y=data_frame2['freq'][:range_idx], name="Predicted Market Share by Product",
                   marker={'color': products_used_data_frame['colours']})
    return {
        'data': [track_down],
        'layout': go.Layout(
            hovermode="closest",
            xaxis={'title': "Car Models", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Number Predicted", 'titlefont': {'color': 'black', 'size': 14, },
                   'tickfont': {'color': 'black'}}
        )
    }



@app.callback(
    Output('product-pie', 'figure'),
    [Input('intermediate-value', 'children')])
def display_pie(jsonified_data):
    
    pred_prod_val  = pd.read_json(jsonified_data,orient ='split')
    data_frame1 = pd.value_counts(pred_prod_val['LiveBrand']).to_frame().reset_index().sort_values('index')
    data_frame1.columns = ['prod ID', 'freq']
    modelX = data_frame1['prod ID']
    colors = ['mediumblue', 'gold', 'teal', 'deepskyblue', 'red', 'purple', 'orange', 'deeppink']
    fig = go.Pie(labels=modelX, values=data_frame1['freq'],textinfo='label+percent',marker=dict(colors=colors))
    
    return {
        
            'data': [fig]   
           
            }


@app.callback(
    Output('intermediate-value-2', 'children'),
    [Input('intermediate-value', 'children')])
def calculateProbs(jsonified_data):
    pred_prod_val = pd.read_json(jsonified_data, orient='split')
    return pred_prod_val.to_json(date_format='iso', orient='split')


@app.callback(
    Output('heatmap', 'figure'),
    [Input('intermediate-value-2', 'children')])
def display_heatmap(jsonified_data):

    nP = data_frame.shape[0]
    ind_provider = np.array([0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7])

    modelX = ['e-tron', 'Bolt', 'I-Pace', 'Niro EV', 'Niro Plg', 'Optima', 'Leaf S', 'Leaf S+', 'Model 3', 'Model S',
            'Model X', 'Prius', 'e-Golf']

    pred_prod_val = pd.read_json(jsonified_data, orient='split')
    per_prod_utility_matrixUse = pred_prod_val.iloc[:, 6:].values
    mat = np.exp(per_prod_utility_matrixUse[:, ~np.all(per_prod_utility_matrixUse == 0, axis=0)])
    scaleVec = mat.sum(axis=1)
    mat = mat / scaleVec[:, None]
    indChoice = mat.argmax(axis=1)

    desirability = np.full((nP, nP), -9.)
    binary = np.full((nP, nP), 0.)
    for i in range(nP):
        if sum(indChoice == i) > 0:
            matProb = mat[indChoice == i, :].T
            for j in range(nP):
                if ind_provider[i] == ind_provider[j]:
                    desirability[i, j] = np.percentile(matProb[np.array(range(nP)) == j, :], 50)
                else:
                    desirability[i, j] = np.percentile(matProb[np.array(range(nP)) == j, :], 80)
                if desirability[i, j] > 0.15:
                    binary[i, j] = desirability[i, j]

    heat = go.Heatmap(z=list(np.round(binary, 4)), x=modelX, y=modelX, colorscale='Viridis')
    return {
        'data': [heat],
        'layout': go.Layout(
            title="Market Share Opportunities",
            hovermode="closest",
            xaxis={'title': "Competitive Market Offers", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},
            yaxis={'title': "Pred Market Offer", 'titlefont': {'color': 'black', 'size': 14},
                   'tickfont': {'size': 9, 'color': 'black'}},

            shapes=[{'type': 'line', 'x0': 0.5, 'y0': -0.5, 'x1': 0.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 1.5, 'y0': -0.5, 'x1': 1.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 2.5, 'y0': -0.5, 'x1': 2.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 5.5, 'y0': -0.5, 'x1': 5.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 7.5, 'y0': -0.5, 'x1': 7.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 10.5, 'y0': -0.5, 'x1': 10.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': 11.5, 'y0': -0.5, 'x1': 11.5, 'y1': 12.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 0.5, 'x1': 12.5, 'y1': 0.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 1.5, 'x1': 12.5, 'y1': 1.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 2.5, 'x1': 12.5, 'y1': 2.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 5.5, 'x1': 12.5, 'y1': 5.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 7.5, 'x1': 12.5, 'y1': 7.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 10.5, 'x1': 12.5, 'y1': 10.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}},
                    {'type': 'line', 'x0': -0.5, 'y0': 11.5, 'x1': 12.5, 'y1': 11.5,
                     'line': {'color': 'rgb(55, 128, 191)', 'width': 3}}
                    ]
        )
    }



@app.callback(
    [Output('brandFig', 'figure'),
     Output('brandWar', 'figure')],
    [Input('intermediate-value-2', 'children'),
     Input('brandBeauty', 'value'),
     Input('brandAttrition', 'value')])
def display_BrandKDE(jsonified_data, typeBrand, whichOne):
    
    pred_prod_val = pd.read_json(jsonified_data, orient='split')
    per_prod_utility_matrixUse = pred_prod_val.iloc[:, 6:].values
    mat = np.exp(per_prod_utility_matrixUse[:, ~np.all(per_prod_utility_matrixUse == 0, axis=0)])
    scaleVec = mat.sum(axis=1)
    mat = mat / scaleVec[:, None]
    probVersion = pred_prod_val.copy()
    probVersion.iloc[:, 6:] = mat
    mat = probVersion.iloc[:, 6:].values
    indChoice = probVersion['LiveProduct'] - 1
    
    ind_provider = np.array([0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7])
    brandInd = [0, 1, 2, 3, 4, 5, 6, 7]
    brandChoice = ind_provider[indChoice]

    labelsSheet = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']
    colors = ['mediumblue', 'gold', 'teal', 'deepskyblue', 'red', 'purple', 'orange', 'deeppink']
    labels = ['Audi', 'Chevrolet', 'Jaguar', 'Kia', 'Nissan', 'Tesla', 'Toyota', 'VW']

    if typeBrand == "Major Brands":
        majorData = [None for x in range(4)]
        iBr = 0
        clrs = [None for x in range(4)]
        for i in [1, 4, 5, 6]:
            matBrand = mat[brandChoice == i, :].T[ind_provider == i, :].T
            matProb = matBrand.sum(axis=1)
            logProb = matProb
            majorData[iBr] = list(logProb)
            clrs[iBr] = colors[i]
            iBr += 1

        brandFigure = ff.create_distplot(majorData, ['Chevrolet', 'Nissan', 'Tesla', 'Toyota'], show_rug=True,
                                         show_hist=False, colors=clrs)
        brandFigure['layout'].update(
            xaxis=dict(title='Prob.(Choose Brand)', tickformat=',.2%'),
            yaxis=dict(title='Density', tickformat=',.3f'),
            title="Major Brands Attraction"
        )
    elif typeBrand == 'Minor Brands':
        minorData = [None for x in range(4)]
        iBr = 0
        clrs = [None for x in range(4)]
        for i in [0, 2, 3, 7]:
            matBrand = mat[brandChoice == i, :].T[ind_provider == i, :].T
            matProb = matBrand.sum(axis=1)
            logProb = matProb
            minorData[iBr] = list(logProb)
            clrs[iBr] = colors[i]
            iBr += 1

        brandFigure = ff.create_distplot(minorData, ['Audi', 'Jaguar', 'Kia', 'VW'], show_rug=True,
                                         show_hist=False, colors=clrs)
        brandFigure['layout'].update(
            xaxis=dict(title='Prob.(Choose Brand)', tickformat=',.2%'),
            yaxis=dict(title='Density', tickformat=',.3f'),
            title="Minor Brands Attraction"
        )

    i = labelsSheet.index(whichOne)
    jRange = brandInd
    brandData = [None for x in jRange]
    clrs = [None for x in jRange]
    jBr = 0
    for j in jRange:
        matBrand = mat[brandChoice == i, :].T[ind_provider == j, :].T
        matProb = matBrand.sum(axis=1)
        
        logProb = matProb
        brandData[jBr] = list(logProb)
        clrs[jBr] = colors[j]
        jBr += 1

    scavengeFigure = ff.create_distplot(brandData, labels, show_rug=True, show_hist=False, colors=clrs)
    scavengeFigure['layout'].update(
        xaxis=dict(title='Prob.(Choose Brand)',
                   tickformat=',.2%'),
        yaxis=dict(title='Density', tickformat=',.3f'),
        title=labelsSheet[i] + ' Attrition'
    )
    return brandFigure, scavengeFigure

if __name__ == '__main__':
    
    app.run_server(debug=True)
