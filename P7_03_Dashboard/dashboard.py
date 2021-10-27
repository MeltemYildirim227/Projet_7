# url dashboard : https://app-dashboard-projet7.herokuapp.com/

import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests


# Load data
train_prep = pd.read_csv('data/train_features.csv')
test_prep = pd.read_csv('data/test_features.csv')
# train_features.csv is obtained by keeping only the columns which is in features list down below of train_prep.csv (create by data_prep.py)
# test_features.csv is obtained by keeping only the columns which is in features list down below and the first 10000 rows of test_prep.csv (create by data_prep.py)


# Plot
colors = {
    'background': '#31302F',
    'text': '#FFFFFF'    
}


# Creates a list of dictionaries, which have the keys 'label' and 'value'
features=['EXT_SOURCE_1',
          'EXT_SOURCE_2',
          'EXT_SOURCE_3',
          'DAYS_BIRTH',
          'AMT_CREDIT',
          'AMT_GOODS_PRICE',
          'DAYS_EMPLOYED',
          'DAYS_ID_PUBLISH',
          'CODE_GENDER_F',
          'DAYS_EMPLOYED_PERCENT',
          'NAME_FAMILY_STATUS_Married']

def get_options(list_features):
    dict_list = []
    for i in list_features:
        dict_list.append({'label': i, 'value': i})
    return dict_list


# Initialize the app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
        children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Prediction of Loan Approval'),
                                 html.P('''Enter Client ID'''),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='idselector',
                                                      options=get_options(test_prep['SK_ID_CURR'].unique()),
                                                      multi=False,
                                                      value=test_prep['SK_ID_CURR'].unique()[0],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='idselector'
                                                      )
                                         ]),
                                 html.P('''Choose the parameter to analyze'''),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='featureselector',
                                                      options=get_options(features),
                                                      multi=False,
                                                      value=features[0],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='featureselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                 ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='fig_result',
                                           ),
                                 dcc.Graph(id='fig',
                                           )
                                 ]
                            )
                              ])
        ]
    )



@app.callback(
    Output('fig_result', 'figure'),
    [Input('idselector', 'value')]
    )
def figure_predict(selected_dropdown_value_id):
    df_test = test_prep.copy()
    client_ids = df_test['SK_ID_CURR'].unique().tolist()
    if selected_dropdown_value_id in client_ids:
        url = 'https://app-flask-projet7.herokuapp.com/predict_api'
        r = requests.post(url, json={'Client ID': selected_dropdown_value_id})
        pred = r.text
        pred = float(pred)
        if pred <= 0.5:
            title='Loan Approved'
        else:
            title='Loan Not Approved'
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=pred,
            title=title,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1],
                           },
                   'bar': {'color': 'black'},
                   'steps': [
                       {'range': [0, 0.5], 'color': 'green'},
                       {'range': [0.5, 1], 'color': 'red'}]
                  }
            ))
        
        fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'])
        return fig           


@app.callback(
    Output('fig', 'figure'),
    [Input('idselector', 'value')],
    [Input('featureselector', 'value')]
    )
def update_graph(selected_dropdown_value_id, selected_dropdown_value_feature):
    df_train = train_prep
    df_test = test_prep
    t_0 = df_train.loc[df_train['TARGET'] == 0, selected_dropdown_value_feature]
    t_1 = df_train.loc[df_train['TARGET'] == 1, selected_dropdown_value_feature]    
    
    data = {'Clients without payment difficulties': t_0,
            'Clients with payment difficulties': t_1
            }
    df = pd.concat(data, axis=1)

    fig = ff.create_distplot([df[c].dropna() for c in df.columns], df.columns,
                             show_hist=False, show_rug=False)
    fig.add_vline(x=df_test.loc[df_test['SK_ID_CURR'] == selected_dropdown_value_id, selected_dropdown_value_feature].item(),
                  line_width=2, line_dash='dash', line_color='red', annotation_text='Client position')
    
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        title_text='Distribution of ' + selected_dropdown_value_feature
        )

    fig['layout']['yaxis'].update(title_text='Density')
    fig['layout']['xaxis'].update(title_text=selected_dropdown_value_feature)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)