# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dash_daq as daq
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook_connected'

# colors
COLOR_SET = px.colors.qualitative.Plotly
COLOR_BAR = ['indianred']

# Reading Dataset
df = pd.read_csv('IMDb_movies.csv')
########################################################################################################
# Modify date type
df.date_published = pd.to_datetime(df.date_published, errors='coerce')
df['Year'] = df.date_published.dt.year

# Drop null values and sort by date
df.dropna(subset=['Year','country'], inplace=True)
df.sort_values('Year', inplace=True)

# convert budget, usa_gross_income and worldwide_gross_income columns to numeric
df.budget = pd.to_numeric(df.budget.str.replace("[^\d]",'').str.strip())
df.usa_gross_income = pd.to_numeric(df.usa_gross_income.str.replace("[^\d]",'').str.strip())
df.worlwide_gross_income = pd.to_numeric(df.worlwide_gross_income.str.replace("[^\d]",'').str.strip())

df['Revenue'] = df.worlwide_gross_income - df.budget

# Extract first genre , country and language
df['Genre'] = pd.DataFrame(df.genre.str.split(',').tolist(), index= df.index)[0]
df['Country'] = pd.DataFrame(df.country.str.split(',').tolist(), index= df.index)[0]

def myFunc(x):
    try:
        x = x.split(',')
        if isinstance(x, list):
            return x[0]
    except:
        return np.nan

df['Language'] = df.language.replace('None', np.nan).apply(myFunc)

# Rename some columns
df.rename(columns={"avg_vote": "Average Vote",
                    'votes':'Vote Count',
                    'budget':'Budget',
                    'duration':'Duration',
                    'title':'Title'},
                    inplace=True)

# filter years
df = df[df['Year'] < 2021]

# select first year
year_filter = df['Year'] == 2015

gy = df.groupby('Year').mean().reset_index()

########################################################################################################
# Create First figure
subfig = make_subplots(specs=[[{"secondary_y": True}]])

# create two independent figures with px.line each containing data from multiple columns
fig = px.line(gy, x='Year', y=['Average Vote'], render_mode="webgl",)
fig2 = px.line(gy, x='Year', y=[ 'Vote Count'], render_mode="webgl",)

fig2.update_traces(yaxis="y2")

subfig.add_traces(fig.data + fig2.data)
subfig.layout.xaxis.title="Year"
subfig.layout.yaxis.title="Vote"
subfig.layout.yaxis2.title="Vote Counts"
# recoloring is necessary otherwise lines from fig und fig2 would share each color
# e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))



########################################################################################################
# Define functions to return figures

def plot_scatter_graph(dataframe, template='ggplot2'):
    return px.scatter(dataframe, x='Budget', y ='Average Vote',
                    size='Vote Count', color = 'Genre',
                    log_x =True, size_max=60,
                    hover_name='Title',hover_data=['Duration','director','language','country'],
                    color_discrete_sequence=COLOR_SET,
                    template=template).update_layout( title={
                            'text': "Average Vote - Budget",
                            'y':0.97,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                    } )
scatter_graph = plot_scatter_graph(df[year_filter])


def plot_hist_duration(dataframe, ):
    return px.histogram(dataframe, x='Duration', color_discrete_sequence=COLOR_BAR).update_layout( title={
                            'text': "Duration Histogram",
                            'y':0.97,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                    } )
hist_duration = plot_hist_duration(df[year_filter])


def plot_treemap(dataframe,  height=600):
    return px.treemap(dataframe, values='Average Vote', path=['Genre', 'Country'], hover_name='Country', height=height, color_discrete_sequence=COLOR_SET)
treemap = plot_treemap(df[year_filter])


def plot_sunbrust(dataframe,  height=600):
    return px.sunburst(dataframe, values='Average Vote', path=['Genre', 'Country'], hover_name='Country', color_discrete_sequence=COLOR_SET).update_layout( title={
                            'text': "Countries per Genres ",
                            'y':0.978,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                    } )
sunbrust = plot_sunbrust(df[year_filter])


def plot_bar_title_length(dataframe):
    counts, bins = np.histogram(dataframe.Title.str.replace(' ', ',').str.split(',').apply(lambda x:len(x)), bins=range(1, 14, 1))
    bins = 0.5 * (bins[:-1] + bins[1:])
    return px.bar(x=bins, y=counts, labels={'x':'Title Length (Words)', 'y':'count'}, color_discrete_sequence=COLOR_BAR)
bar_title_length = plot_bar_title_length(df[year_filter])


def plot_bar_language(dataframe,  height=600):
    lang_df = dataframe.groupby('Language').size().sort_values(ascending=False).reset_index(name='counts')
    return px.bar(lang_df[:10], x='Language', y='counts',  barmode='group', log_y=True, color_discrete_sequence=COLOR_BAR)
bar_language = plot_bar_language(df[year_filter])


def plot_bar_genre(dataframe, ):
    genre_df = dataframe.groupby('Genre').size().sort_values(ascending=False).reset_index(name='counts')
    return px.bar(genre_df[genre_df.counts >5], x='Genre', y='counts', color='Genre', color_discrete_sequence=COLOR_SET).update_layout(showlegend=False)
bar_genre = plot_bar_genre(df[year_filter])


def plot_bar_country(dataframe, ):
    country_df = dataframe.groupby(by=['country']).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)[:10]
    return px.bar(data_frame=country_df, x='country', y='counts', barmode='group', log_y=True, color_discrete_sequence=COLOR_BAR)
bar_country = plot_bar_country(df[year_filter])

########################################################################################################
cg = df[year_filter].groupby('Country').mean().reset_index()

# add iso code to dataframe to plot countries on map
import pycountry

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3


un_iso = {'Bolivia': 'BOL',
        'Brunei':	'BRN' ,
        'Czech Republic' : 'CZE'  ,
        'Czechoslovakia':'CZE'  ,
        'East Germany':	'DEU'  ,
        'Federal Republic of Yugoslavia': 'YUG' ,
        'Iran' : 'IRN',
        'Isle Of Man': 'IMN' ,
        'Korea': 'KOR',
        'Kosovo': 'XXK',
        'Laos': 'LAO',
        'Moldova': 'MDA',
        'Netherlands Antilles': 'ANT',
        'North Korea':'PRK' ,
        'North Vietnam': 'VDR'  ,
        'Palestine': 'PSE' ,
        'Republic of North Macedonia' : 'MKD' ,
        'Russia': 'RUS',
        'Serbia and Montenegro': 'SCG' ,
        'South Korea': 'KOR' ,
        'Soviet Union': 'SUN'  ,
        'Syria': 'SYR',
        'Taiwan': 'zho',
        'Tanzania': 'TZA' ,
        'The Democratic Republic Of Congo': 'COD',
        'UK': 'GBR',
        'USA': 'USA',
        'Venezuela': 'VEN' ,
        'Vietnam': 'VNM',
        'West Germany' :'DEU'  ,
        'Yugoslavia': 'YUG' }

countries.update(un_iso)
codes = [countries.get(country, 'Unknown code') for country in cg.Country]
cg['iso_code'] = codes


def plot_choropleth_map(dataframe, title=None, height=600):
    # check-me MAX
    cg = dataframe.groupby('Country').max().reset_index()
    cg['iso_code'] = [countries.get(country, 'Unknown code') for country in cg.Country]

    return px.choropleth(cg, color ='Average Vote', locations='iso_code', hover_name='Country').update_layout(
                            title={
                            'text': "Maximum Vote for each Country",
                            'y':0.97,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}, 
                            geo=dict(bgcolor= 'rgba(0,0,0,0)'),
                            coloraxis = {'colorbar':{'title': 'Vote'}})

# df2 = df[['Country', 'Average Vote']]
choropleth_map = plot_choropleth_map(df[year_filter])


# ===============================================================================================================
# # DASH 
# ===============================================================================================================


import dash
from jupyter_dash import JupyterDash
from dash import html
from dash import dcc
import plotly.express as px
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

#*********************************************************************************************************************
# Draw figure Function
def drawFigure(figure=None, id=None, config={'displayModeBar': False, 'autosizable':True, 'responsive':True}):
    return dcc.Graph(
                    id=id,
                    figure=figure.update_layout(
                        template     = 'plotly_dark',
                        plot_bgcolor = 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize=True,
                        margin=dict(l=20, r=20, t=40, b=20),
                    ),
                    config=config,
                    style={'background-color':'#32383E', 'height':'100%'}
                )


# Text field Function
def drawText(text='My Text', style={'textAlign': 'center'}):
    return html.Div([
        html.H1(text, style={
                'flex':'0', 
                'margin-bottom':'36px',
                'font-size': '45px',
                'color': 'white'
            }),
    ], style={'padding-top':'18px'})

#*********************************************************************************************************************
# Build App
app = JupyterDash(external_stylesheets=[dbc.themes.SLATE])

# Custom Style
NAVBAR_STYLE = {
    'height': "200px",
}

ROW_STYLE = {
    'height': "450px",
}

TITLE_STYLE={
    'height': '130px', 
    'flex':'0',
    'margin':'0 auto',
    'textAlign': 'center',
    'margin-bottom':'10px'
}

#*********************************************************************************************************************
import base64
image_filename = 'logo4.png' 
encoded_image  = base64.b64encode(open(image_filename, 'rb').read()).decode("utf-8")

#*********************************************************************************************************************
app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            # ------------------------------------------ Bans Section  ------------------------------------------ #
            dbc.Row([
                dbc.Col([
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image), height=120, style={'margin-bottom':'20px'}),
                    html.H1("Internet Movie Database", style={'font-size': '20px','color': 'white'}),
                    daq.Slider(
                        id='slider_year',
                        min=df['Year'].min(),
                        max=df['Year'].max(),
                        step=None,
                        value=2015,
                        color='#bd9321',
                        marks = { str(i):str(i) for i in df['Year'].unique() },
                        handleLabel={"label": "Year", "showCurrentValue": True, 'style':{'height':'33px', 'top':'20px', 'font-size':'10px'}},
                        labelPosition='top'
                    ),
                    html.Div(id='slider-output-container')
                ], width=2, style={'text-align': 'center'}),
                #===========##===========##===========##===========##===========#
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("World Wide Gross", style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                            html.H1("${:,.0f}M".format(df[year_filter]['worlwide_gross_income'].sum()/1000000), id='gross', style={'font-size': '30px','color': 'white'}),
                        ], width=4, style={"height": "180px", 'font-family': 'Cursive', "background-color": "#37597d", 'text-align':'center', 'justify':'center', 'align':'center', 'padding-top': '40px'}),

                        dbc.Col([
                            html.H6("Total Budget", style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                            html.H1("${:,.0f}M".format(df[year_filter]['Budget'].sum()/1000000), id='budget', style={'font-size': '30px','color': 'white'}),
                        ], width=4, style={"height": "180px", 'font-family': 'Cursive', "background-color": "#e37900", 'text-align':'center', 'justify':'center', 'align':'center', 'padding-top': '40px'}),
                        
                        dbc.Col([
                            html.H6("Year", style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                            html.H1("2015", id='year', style={'font-size': '40px','color': 'white'}),
                        ], width=4, style={"height": "180px", 'font-family': 'Cursive', "background-color": "#e24b3c", 'text-align':'center', 'justify':'center', 'align':'center', 'padding-top': '40px'}),
                    ]),
                ], width=4), 
                #===========##===========##===========##===========##===========#
                dbc.Col([
                    # drawFigure(id='subfig' ,figure=subfig)
                    dcc.Graph(
                    id='subfig',
                    figure=subfig.update_layout(
                        template     = 'plotly_dark',
                        plot_bgcolor = 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        autosize=True,
                        margin=dict(l=20, r=20, t=20, b=20),
                    ),
                    config={'displayModeBar': False, 'autosizable':True, 'responsive':True},
                    style={'background-color':'#32383E', 'height':'100%'}
                ) ], width=6, style={"height": "180px"}),
            ], align='center', style= NAVBAR_STYLE),

            html.Br(),
            # ------------------------------------------ Graph Section 1  ------------------------------------------ #
            dbc.Row([
                dbc.Col([
                    # dcc.Graph(id='scatter_graph')
                    drawFigure(id='scatter_graph' ,figure=scatter_graph)
                ], width=9,  style={"height": "100%"}),
                #===========##===========##===========##===========##===========#
                dbc.Col([
                    drawFigure(id='hist_duration' ,figure=hist_duration)
                ], width=3,  style={"height": "100%"}),
            ], align='center', style=ROW_STYLE),

            html.Br(),
            # ------------------------------------------ Graph Section 2  ------------------------------------------ #
            dbc.Row([
                dbc.Col([
                    dbc.Tabs(
                        [
                            dbc.Tab( label="Genre", children=[drawFigure(id='bar_genre', figure=bar_genre)]),
                            dbc.Tab( label="Country", children=[drawFigure(id='bar_country', figure=bar_country)]),
                            dbc.Tab( label="Language", children=[drawFigure(id='bar_language', figure=bar_language)]),
                            dbc.Tab( label="Title Length", children=[drawFigure(id='bar_title_length', figure=bar_title_length)]),
                        ], style={'height':'25px', 'position':'absolute', 'right':'50px','z-index':'5'}
                    )
                ], width=3, style={"height": "100%", 'position':'relative'}),
                #===========##===========##===========##===========##===========#
                dbc.Col([
                    drawFigure(id='sunbrust' ,figure=sunbrust)
                ], width=3,  style={"height": "100%"}),
                #===========##===========##===========##===========##===========#
                dbc.Col([
                    drawFigure(id='choropleth_map', figure=choropleth_map)
                ], width=6,  style={"height": "100%"}),
            ], align='center', style=ROW_STYLE),
            
            html.Br(),
            # ------------------------------------------ About Section   ------------------------------------------ #
            dbc.Row([
                dbc.Col([

                ], width=3,  style={"height": "100%", }),
                dbc.Col([
                    html.H3("Developers", style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                ], width=2,  style={"height": "100%", }),
                dbc.Col([
                    html.A(['Ahmed N. Awaad'], href='https://www.linkedin.com/in/ahmed-n-awaad/',target='_blank', style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                ], width=2,  style={"height": "100%", }),
                dbc.Col([
                    html.A(['Mohamed ELMesawy'], href='https://www.linkedin.com/in/mohamed-elmesawy/',target='_blank', style={'font-size': '20px','color': 'white', 'font-family':'Sans-serif'}),
                ], width=2,  style={"height": "100%", }),
                #===========##===========##===========##===========##===========#
            ], align='center',  style={"height": "30px"}),
        ]), color = 'dark'
    )
])
#*********************************************************************************************************************

@app.callback(
    Output(component_id='scatter_graph', component_property='figure'),
    Output(component_id='choropleth_map', component_property='figure'),
    Output(component_id='sunbrust', component_property='figure'),
    Output(component_id='hist_duration', component_property='figure'),
    Output(component_id='bar_genre', component_property='figure'),
    Output(component_id='bar_country', component_property='figure'),
    Output(component_id='bar_language', component_property='figure'),
    Output(component_id='bar_title_length', component_property='figure'),

    Output(component_id='gross', component_property='children'),
    Output(component_id='budget', component_property='children'),
    Output(component_id='year', component_property='children'),
    
    Input(component_id='slider_year', component_property='value')
)
def update_graph_by_year_slider(input_slider):
    # generate filtered dataframe
    filtered_dataframe = df[df['Year'] == float(input_slider)]

    # Create plots
    scatter_graph    = plot_scatter_graph(filtered_dataframe)
    choropleth_map   = plot_choropleth_map(filtered_dataframe)
    sunbrust         = plot_sunbrust(filtered_dataframe)
    hist_duration    = plot_hist_duration(filtered_dataframe)
    bar_genre        = plot_bar_genre(filtered_dataframe)
    bar_country      = plot_bar_country(filtered_dataframe)
    bar_language     = plot_bar_language(filtered_dataframe)
    bar_title_length = plot_bar_title_length(filtered_dataframe)

    # Iterate over new plots and update layout
    graphs_list = [scatter_graph, choropleth_map, sunbrust, hist_duration, bar_genre, bar_country, bar_language, bar_title_length]
    for graph in graphs_list:
        graph.update_layout(
            template     = 'plotly_dark',
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor= 'rgba(0, 0, 0, 0)',
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20),
            font_family="Sans-serif",
            font_color="white",
            title_font_family="Sans-serif",
            title_font_color="white",
            legend_title_font_color="white"
        )

    # Update Bans
    budget = filtered_dataframe['Budget'].sum()/1000000
    gross = filtered_dataframe['worlwide_gross_income'].sum()/1000000

    graphs_list += ["${:,.0f}M".format(gross),  "${:,.0f}M".format(budget), str(input_slider)]

    return  tuple(graphs_list)

#*********************************************************************************************************************
app.run_server(port=9090)





