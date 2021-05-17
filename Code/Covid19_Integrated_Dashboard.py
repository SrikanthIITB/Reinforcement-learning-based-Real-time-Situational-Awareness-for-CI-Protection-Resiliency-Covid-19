
import time

import numpy as np
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import base64
from MADDPG_QUERY import findactions


df = pd.read_csv("C:\\Users\\catch\\Downloads\\mumbai_hospitals.csv")
df_col = pd.read_csv("C:\\Users\\catch\\Downloads\\mumbai_hospitals.csv").ward.unique()
grouped = df.groupby('ward').sum().reset_index()
ward_n=grouped[['ward','lat']]
grouped=grouped[['ward','Bed Capacity']]
#grouped['Capacity']=grouped['count'].agg([np.sum])
ward_name=pd.DataFrame(df_col, columns = ['Ward Name'])

w1_data=df[df['ward']=='Ward 1']
w1_data=w1_data[['Hospital Name','Bed Capacity']]
w2_data=df[df['ward']=='Ward 2']
w2_data=w2_data[['Hospital Name','Bed Capacity']]
w3_data=df[df['ward']=='Ward 3']
w3_data=w3_data[['Hospital Name','Bed Capacity']]

########

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.models import load_model


df_confirmed = pd.read_csv("C:\\Users\\catch\\Downloads\\Daily data of 3 districts of Mumbai - Mumbai.csv")

df_confirmed=df_confirmed.drop(['State','District','Tested'],axis=1)
df_confirmed.set_index("Date", inplace = True)
df_confirmed.index = pd.to_datetime(df_confirmed.index, format = '%Y-%m-%d')
df_confirmed_country=df_confirmed
df_confirmed_country=df_confirmed_country.drop(['Recovered','Deceased'],axis=1)
df_confirmed_country
#Use data until 14 days before as training
x = len(df_confirmed_country)-14
train=df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

scaler = MinMaxScaler()
scaler.fit(train) 

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

#Sequence size has an impact on prediction, especially since COVID is unpredictable!
seq_size = 7  ## number of steps (lookback)
n_features = 1 ## number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = seq_size, batch_size=1)

#Check data shape from generator
x,y = train_generator[10]

#Also generate test data
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
#Check data shape from generator
x,y = test_generator[0]


prediction = [] #Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:] #Final data points in train 
current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

## Predict future, beyond test dates
model = load_model("C:\\Users\\catch\\Downloads\\MTP 2021\\model_load\\W1_trend_model",compile = False)

future = 7 #Days
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

### Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index

#Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

#Create a dataframe to capture the forecast data
df_forecast1 = pd.DataFrame(columns=["actual_confirmed","predicted"], index=time_series_array)
df_forecast1.loc[:,"predicted"] = rescaled_prediction[:,0]
df_forecast1.loc[:,"actual_confirmed"] = test["Confirmed"]
df_forecast1['Date']=df_forecast1.index

import plotly.graph_objects as go

trend1 = go.Figure()
trend1.add_trace(go.Scatter(x=df_forecast1['Date'], y=df_forecast1['predicted'],
                    mode='lines',
                    name='predicted'))
trend1.add_trace(go.Scatter(x=df_forecast1['Date'], y=df_forecast1['actual_confirmed'],
                    mode='lines+markers',
                    name='actual_confirmed'))
trend1.update_layout(margin=dict(
                    r=1, l=1,
                    b=1, t=1))

#df_forecast1.to_csv('file1.csv')



df_confirmed = pd.read_csv("C:\\Users\\catch\\Downloads\\Daily data of 3 districts of Mumbai - Pune.csv")

df_confirmed=df_confirmed.drop(['State','District','Tested'],axis=1)
df_confirmed.set_index("Date", inplace = True)
df_confirmed.index = pd.to_datetime(df_confirmed.index, format = '%Y-%m-%d')
df_confirmed_country=df_confirmed
df_confirmed_country=df_confirmed_country.drop(['Recovered','Deceased'],axis=1)
df_confirmed_country
#Use data until 14 days before as training
x = len(df_confirmed_country)-14
train=df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

scaler = MinMaxScaler()
scaler.fit(train) 

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

#Sequence size has an impact on prediction, especially since COVID is unpredictable!
seq_size = 7  ## number of steps (lookback)
n_features = 1 ## number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = seq_size, batch_size=1)

#Check data shape from generator
x,y = train_generator[10]

#Also generate test data
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
#Check data shape from generator
x,y = test_generator[0]


prediction = [] #Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:] #Final data points in train 
current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

## Predict future, beyond test dates
model = load_model("C:\\Users\\catch\\Downloads\\MTP 2021\\model_load\\W2_trend_model",compile = False)

future = 7 #Days
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

### Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index

#Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

#Create a dataframe to capture the forecast data
df_forecast2 = pd.DataFrame(columns=["actual_confirmed","predicted"], index=time_series_array)
df_forecast2.loc[:,"predicted"] = rescaled_prediction[:,0]
df_forecast2.loc[:,"actual_confirmed"] = test["Confirmed"]
df_forecast2['Date']=df_forecast2.index

import plotly.graph_objects as go

trend2 = go.Figure()
trend2.add_trace(go.Scatter(x=df_forecast2['Date'], y=df_forecast2['predicted'],
                    mode='lines',
                    name='predicted'))
trend2.add_trace(go.Scatter(x=df_forecast2['Date'], y=df_forecast2['actual_confirmed'],
                    mode='lines+markers',
                    name='actual_confirmed'))
trend2.update_layout(margin=dict(
                    r=1, l=1,
                    b=1, t=1))


df_confirmed = pd.read_csv("C:\\Users\\catch\\Downloads\\Daily data of 3 districts of Mumbai - Thane.csv")

df_confirmed=df_confirmed.drop(['State','District','Tested'],axis=1)
df_confirmed.set_index("Date", inplace = True)
df_confirmed.index = pd.to_datetime(df_confirmed.index, format = '%Y-%m-%d')
df_confirmed_country=df_confirmed
df_confirmed_country=df_confirmed_country.drop(['Recovered','Deceased'],axis=1)
df_confirmed_country
#Use data until 14 days before as training
x = len(df_confirmed_country)-14
train=df_confirmed_country.iloc[:x]
test = df_confirmed_country.iloc[x:]

scaler = MinMaxScaler()
scaler.fit(train) 

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

#Sequence size has an impact on prediction, especially since COVID is unpredictable!
seq_size = 7  ## number of steps (lookback)
n_features = 1 ## number of features. This dataset is univariate so it is 1
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = seq_size, batch_size=1)

#Check data shape from generator
x,y = train_generator[10]

#Also generate test data
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=seq_size, batch_size=1)
#Check data shape from generator
x,y = test_generator[0]


prediction = [] #Empty list to populate later with predictions

current_batch = train_scaled[-seq_size:] #Final data points in train 
current_batch = current_batch.reshape(1, seq_size, n_features) #Reshape

## Predict future, beyond test dates
model = load_model("C:\\Users\\catch\\Downloads\\MTP 2021\\model_load\\W3_trend_model",compile = False)

future =7 #Days
for i in range(len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

### Inverse transform to before scaling so we get actual numbers
rescaled_prediction = scaler.inverse_transform(prediction)

time_series_array = test.index

#Add new dates for the forecast period
for k in range(0, future):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

#Create a dataframe to capture the forecast data
df_forecast3 = pd.DataFrame(columns=["actual_confirmed","predicted"], index=time_series_array)
df_forecast3.loc[:,"predicted"] = rescaled_prediction[:,0]
df_forecast3.loc[:,"actual_confirmed"] = test["Confirmed"]
df_forecast3['Date']=df_forecast3.index

import plotly.graph_objects as go

trend3 = go.Figure()
trend3.add_trace(go.Scatter(x=df_forecast3['Date'], y=df_forecast3['predicted'],
                    mode='lines',
                    name='predicted'))
trend3.add_trace(go.Scatter(x=df_forecast3['Date'], y=df_forecast3['actual_confirmed'],
                    mode='lines+markers',
                    name='actual_confirmed'))
trend3.update_layout(margin=dict(
                    r=1, l=1,
                    b=1, t=1))

slope1=abs(df_forecast1['predicted'][-1]-df_forecast1['predicted'][-future])/future
slope1=grouped['Bed Capacity'][0]/slope1
slope2=abs(df_forecast2['predicted'][-1]-df_forecast2['predicted'][-future])/future
slope2=grouped['Bed Capacity'][1]/slope2
slope3=abs(df_forecast3['predicted'][-1]-df_forecast3['predicted'][-future])/future
slope3=grouped['Bed Capacity'][2]/slope3

ci_status = pd.read_csv("C:\\Users\\catch\\Downloads\\CI_status.csv")

def findstatus(w):
    #status=ci_status[ci_status['Ward']==w]
    status=w
    c_status='secondary'
    if (status==1).bool():
        c_status='success'
    if (status==2).bool():
        c_status='warning'
    if (status==3).bool():
        c_status='danger'
    return c_status

def h_status(w):
    #status=ci_status[ci_status['Ward']==w]
    status=w
    c_status='secondary'
    #print('status: ',status)
    if (status>0.20):
        c_status='success'
    elif (status>0.15):
        c_status='warning'
    else:
        c_status='danger'
    return c_status

def hi_status(w):
    #status=ci_status[ci_status['Ward']==w]
    status=w
    #print('status: ',status)
    if (status>0.20):
        i_status=1
    elif (status>0.15):
        i_status=2
    else:
        i_status=3
    return i_status


all_actions = []

w1_all_actions1 = findactions(int(ci_status[ci_status['Ward']==1]['Power']),int(ci_status[ci_status['Ward']==1]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w1_all_actions2 = findactions(hi_status(slope1),int(ci_status[ci_status['Ward']==1]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w1_all_actions3 = findactions(int(ci_status[ci_status['Ward']==1]['Power']),hi_status(slope1), hi_status(slope1), hi_status(slope2), hi_status(slope3) )

print('w1_all_actions1',w1_all_actions1,":",w1_all_actions2,":",w1_all_actions2)

w2_all_actions1 = findactions(int(ci_status[ci_status['Ward']==2]['Power']),int(ci_status[ci_status['Ward']==2]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w2_all_actions2 = findactions(hi_status(slope2),int(ci_status[ci_status['Ward']==2]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w2_all_actions3 = findactions(int(ci_status[ci_status['Ward']==2]['Power']),hi_status(slope2), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
print('w1_all_actions2',w2_all_actions1,":",w2_all_actions2,":",w2_all_actions3)
w3_all_actions1 = findactions(int(ci_status[ci_status['Ward']==3]['Power']),int(ci_status[ci_status['Ward']==3]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w3_all_actions2 = findactions(hi_status(slope3),int(ci_status[ci_status['Ward']==3]['Transport']), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
w3_all_actions3 = findactions(int(ci_status[ci_status['Ward']==3]['Power']),hi_status(slope3), hi_status(slope1), hi_status(slope2), hi_status(slope3) )
print('w1_all_actions3',w3_all_actions1,":",w3_all_actions2,":",w3_all_actions3)

h_actions = {
    0 : "Turn on backup electricity generators for medical equiments",
    1: "Keep the ambulances,vehicles ready for the incoming patients",
    2: "Increase the number of Beds for the incoming patients",
    3: "No actions required to be taken for Hospital CI"
 }

e_actions = {
    0 : "Power incoming is at critical levels due to surge in current demand",
    1: "Alert maintenance vehicles about alternative routes",
    2: "Take measures to increase power output for future demand",
    3: "No actions required to be taken for Power CI"
}

t_actions = {
    0 : "Turn on backup electricity generators due to potential powercuts",
    1: "Share Alternative routes with all ambulances of the ward",
    2: "Alert! Potential transport-blockages due to Covid restrictions",
    3: "No action required to be taken for Transport CI "
  }
result=['','','','','','','','']
if slope1 >= slope2 and slope1 >= slope3:
      result[0]="Move the incoming patients to Hostipal in Ward 1"
      result[1]=h_actions[w1_all_actions1[0]]
      result[2]=e_actions[w1_all_actions2[1]]
      result[3]=t_actions[w1_all_actions3[2]]
      result[4]=h_status(slope1)
      result[5]=findstatus(ci_status[ci_status['Ward']==1]['Power'])
      result[6]=findstatus(ci_status[ci_status['Ward']==1]['Transport'])
elif slope2 >= slope1 and slope2 >= slope3:
      result[0]="Move the incoming patients to Hostipal in Ward 2"
      result[1]=h_actions[w2_all_actions1[0]]
      result[2]=e_actions[w2_all_actions2[1]]
      result[3]=t_actions[w2_all_actions3[2]]
      result[4]=h_status(slope2)
      result[5]=findstatus(ci_status[ci_status['Ward']==2]['Power'])
      result[6]=findstatus(ci_status[ci_status['Ward']==2]['Transport'])
elif slope3 >= slope1 and slope3 >= slope3:
      result[0]="Move the incoming patients to Hostipal in Ward 3"
      result[1]=h_actions[w3_all_actions1[0]]
      result[2]=e_actions[w3_all_actions2[1]]
      result[3]=t_actions[w3_all_actions3[2]]
      result[4]=h_status(slope3)
      result[5]=findstatus(ci_status[ci_status['Ward']==3]['Power'])
      result[6]=findstatus(ci_status[ci_status['Ward']==3]['Transport'])

label_3d='3D view of Hospital Capacities in Mumbai City'

################################################################################

w1_map = px.scatter_mapbox(df[df.ward=="Ward 1"], lat="lat", lon="long",text='ward',
                           center=dict(lat=19.044960947093976, lon=72.88640296258399),
                           color_discrete_sequence=['Orange'], hover_name='Hospital Name',size='Bed Capacity',
                           hover_data={'Hospital Name':False,'Bed Capacity':True,'ward':True,'lat':False,'long':False},
                           zoom=10.5)
#w1_map.update_layout(uniformtext_minsize=18, uniformtext_mode='hide')
w1_map.update_layout(
    font_family="Courier New",
    font_color="White",
    font_size=15,
)

w1_map.layout.showlegend=False
w1_map.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')
w1_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

w2_map = px.scatter_mapbox(df[df.ward=="Ward 2"], lat="lat", lon="long",text='ward',center=dict(lat=19.044960947093976, lon=72.88640296258399),
                           color_discrete_sequence=['MediumPurple'], hover_name='Hospital Name',size='Bed Capacity',
                           hover_data={'Hospital Name':False,'Bed Capacity':True,'ward':True,'lat':False,'long':False},
                           zoom=10.5)

w2_map.update_layout(
    font_family="Courier New",
    font_color="White",
    font_size=15,
)

w2_map.layout.showlegend=False
w2_map.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')
w2_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

w3_map = px.scatter_mapbox(df[df.ward=="Ward 3"], lat="lat", lon="long",text='ward',center=dict(lat=19.044960947093976, lon=72.88640296258399),
                           color_discrete_sequence=['Crimson'], hover_name='Hospital Name',size='Bed Capacity',
                           hover_data={'Hospital Name':False,'Bed Capacity':True,'ward':True,'lat':False,'long':False},
                           zoom=10.5)
w3_map.update_layout(
    font_family="Courier New",
    font_color="White",
    font_size=15,
)

w3_map.layout.showlegend=False
w3_map.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')
w3_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
base_map = px.scatter_mapbox(df,lat="lat", lon="long", center=dict(lat=19.044960947093976, lon=72.88640296258399),
                           color="ward", hover_name='Hospital Name',size='Bed Capacity',text='ward',
                           color_discrete_sequence=['Orange','MediumPurple','Crimson'],
                           hover_data={'Hospital Name':False,'Bed Capacity':True,'ward':True,'lat':False,'long':False},
                           zoom=10.5)
#base_map.update_layout(hoverlabel=dict(font_size=14,font_family="Rockwell"))
base_map.update_layout(
    font_family="Courier New",
    font_color="White",
    font_size=15,
)
base_map.layout.showlegend=False
base_map.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')
base_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

df_heat = pd.read_csv("C:/Users/catch/Downloads/mumbai_hospitals_3days.csv")

fig_heat = px.density_mapbox(df_heat, lat='lat', lon='long', z='CI Health Index', radius=df_heat['count']/8,animation_frame="day",animation_group='ward',
                        center=dict(lat=19.044960947093976, lon=72.88640296258399), zoom=10,
                        hover_data={'name':True,'day':False,'count':False,'CI Health Index':True,'ward':True,'lat':False,'long':False},
                        range_color=(0,1),template="plotly_dark",
                        color_continuous_midpoint=250,color_continuous_scale='thermal_r')
fig_heat.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')
#fig.update(layout_coloraxis_showscale=False)
#fig.show()
#fig_heat.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
#fig_heat.layout.showlegend=False
fig_heat.update_layout(margin={"r":0,"t":0,"l":10,"b":0})


import plotly.graph_objects as go
import numpy as np

# Define random surface
plotfig = go.Figure()

maddpg_fig= go.Figure()
maddpg_fig.add_trace(
    go.Scatter(
        mode="markers",
        marker_opacity=0,
        x=[0, 500],
        y=[0, 500]
    )
)
maddpg_fig.update_xaxes(visible=False)
maddpg_fig.update_yaxes(visible=False)
#maddpg_fig.add_layout_image(dict(source="C:/Users/catch/OneDrive/Pictures/Sysmex_offer.PNG"))

# Add image
maddpg_fig.add_layout_image(
    dict(
        x=0,
        sizex=500,
        y=500,
        sizey=500,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source="https://raw.githubusercontent.com/SrikanthIITB/Reinforcement-learning-based-Real-time-Situational-Awareness-for-CI-Protection-Resiliency-Covid-19/e5d3eada9f2d14c0e65b0018f6c72c9746d06934/Code/Model/MADDPG_REWARDS.PNG")
)
maddpg_fig.update_layout(
    width=730,
    height=350,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)




dfs = pd.read_csv("C:\\Users\\catch\\Downloads\\mumbai_hospitals.csv")
xx1=df[df['ward']=='Ward 1']['lat']
yy1=df[df['ward']=='Ward 1']['long']
zz1=df[df['ward']=='Ward 1']['Bed Capacity']

xx2=df[df['ward']=='Ward 2']['lat']
yy2=df[df['ward']=='Ward 2']['long']
zz2=df[df['ward']=='Ward 2']['Bed Capacity']

xx3=df[df['ward']=='Ward 3']['lat']
yy3=df[df['ward']=='Ward 3']['long']
zz3=df[df['ward']=='Ward 3']['Bed Capacity']
plotfig.add_trace(go.Mesh3d(name="Ward 1",x=xx1,
                   y=yy1,
                   z=zz1,
                   opacity=0.5,
                   color='red'
                  ))
plotfig.add_trace(go.Mesh3d(name="Ward 2",x=xx2,
                   y=yy2,
                   z=zz2,
                   opacity=0.5,
                   color='green'
                  ))
plotfig.add_trace(go.Mesh3d(name="Ward 3",x=xx3,
                   y=yy3,
                   z=zz3,
                   opacity=0.5,
                   color='blue'
                  ))
plotfig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white"),
                    
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),
                    
                    xaxis_title='Latitude',
                    yaxis_title='Longitude',
                    zaxis_title='Capacity of Hospitals'),
    margin=dict(r=1, l=1,b=1, t=1))

#plotfig.show()


#per_ward=pd.concat([ward_name, groupedd['count']],join = 'inner', axis=1)
#per_ward=ward_name.join(groupedd['count'])

df_col=list(df_col)

new_df=df[df.ward=="Ward 1"]

#df = pd.read_csv("https://docs.google.com/spreadsheets/d/1NEThnf5-qkUvsWCy55r5MrI5oQYNzKnBnHbu5a1BLKs/edit?usp=sharing")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Define app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}], external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#map_fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="City", hover_data=["State", "Population"],color_discrete_sequence=["red"], zoom=6)
map_fig = px.scatter_mapbox(df, lat="lat", lon="long",text='Hospital Name',center=dict(lat=19.044960947093976, lon=72.88640296258399),
                            color="ward", hover_name='Hospital Name',size='Bed Capacity',
                            hover_data={'Hospital Name':False,'Bed Capacity':True,'ward':True,'lat':False,'long':False},
                            zoom=10.5)
map_fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)
#map_fig.update_layout(coloraxis_showscale=False)
#map_fig.layout.update(margin={showlegend=False)
#map_fig.update_traces(marker_showscale=False)
map_fig.layout.showlegend=False
#map_fig.layout.update(showlegend=False)
map_fig.update_layout(mapbox_style="dark",mapbox_accesstoken='pk.eyJ1Ijoic3Jpa2FudGhpaXRiIiwiYSI6ImNrNGpkaHpjbTAxbXkzZW1vcWN0OGo5aGUifQ.6ExPnK8GeLZoBeMPaiQjfw')

map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



card_content1 = [
    dbc.CardHeader("Hospital",style={'fontSize':20}),
    dbc.CardBody(
        [
            html.H5("Action", className="card-title"),
            html.Div(id="action1",style={'fontWeight': 'bold','font_size':'20'})
        ]
    ),
]
card_content2 = [
    dbc.CardHeader("Transportation",style={'fontSize':20}),
    dbc.CardBody(
        [
            html.H5("Action", className="card-title"),
            html.Div(id="action2",style={'fontWeight': 'bold','font_size':'20'})
        ]
    ),
    ]
card_content3 = [
    dbc.CardHeader("Power Station",style={'fontSize':20}),
    dbc.CardBody(
        [
            html.H5("Action", className="card-title"),
            html.Div(id="action3",style={'fontWeight': 'bold','font_size':'20'})
        ]
    ),
]
cards = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(card_content1,id="Agent1",color="secondary", inverse=True,style={"margin-top": 10})),
                dbc.Col(dbc.Card(card_content2,id="Agent2", color="secondary", inverse=True,style={"margin-top": 10})),
                dbc.Col(dbc.Card(card_content3,id="Agent3", color="secondary", inverse=True,style={"margin-top": 10})),
            ],
            className="mb-4",
        )
    ]
)

controls = dbc.Card(
    [
       dbc.FormGroup(
            [
                #dbc.Label("Details"),
                dcc.Dropdown(
                    id = "drop_input",
                    placeholder='Choose a Ward...',
                    options=[{'label': i, 'value': i} for i in df.ward.unique()],
                    value='')                    
            ]
        ),
        dbc.FormGroup([dash_table.DataTable(id='table',row_selectable=False,
                                            style_cell={'textAlign': 'center','height': 'auto',
                                                        'minWidth': '80px',
                                                        'width': '80px',
                                                        'maxWidth': '80px',
                                                        'whiteSpace': 'normal'},
                                            style_header={'backgroundColor': 'rgb(230, 230, 230)',
                                                          'fontWeight': 'bold'},
                                            columns=[{"name": i, "id": i} for i in grouped.columns],
                                            data=grouped.to_dict('records'))]),
    
        dbc.FormGroup(
            [
                 dcc.Loading(id="loading-1",color='white',
                             children=[
                                 dbc.Spinner(
                    [
                        dbc.Button("Run Simulation", id="button-run",n_clicks=0)
                    ]
                ),html.Br(),html.Br(),

                html.Div([html.Div(id="output",
                                      style={'font-weight': 'bold','backgroundColor': '#1fae51','color': 'white','height': '30px','fontSize': 20})])],
                             type="default")
                                 
                 #html.Div(id="output",style={'font-weight': 'bold','font_size':'50px','backgroundColor': '#EAEAEA','color': 'black'}),
                 #dcc.Loading(id="loading-1",type="graph",color='green')

            ]
        ),
    ],
    body=True,
    style={'height':'330px','background': '#7FDBFF'},
)
w1_image = 'C:/Users/catch/Downloads/w1.png' # replace with your own image
w1_encoded_image = base64.b64encode(open(w1_image, 'rb').read())
w2_image = 'C:/Users/catch/Downloads/w2.png' # replace with your own image
w2_encoded_image = base64.b64encode(open(w2_image, 'rb').read())
w3_image = 'C:/Users/catch/Downloads/w3.png' # replace with your own image
w3_encoded_image = base64.b64encode(open(w3_image, 'rb').read())

# Define Layout
app.layout = dbc.Container(
    fluid=True,
        style={'background': '#DAF7A6','textAlign': 'center'},
    children=[
        html.H1("Covid-19 Critical Infrastructure Resilience Simulation", style={'textAlign': 'center',}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    width=5,
                    children=[
                        controls,
                        dbc.Card(
                            body=True,
                                style={'height':'450px',"background": "#7FDBFF","margin-top": 5},
                            children=[
                                dbc.FormGroup(
                                    [
                                        html.Div(id="plot_label",style={'font-weight': 'bold','fontSize':20}),
                                        html.Br(),
                #dbc.Label("3D view of Hospital Capacities in Mumbai City"),
                dcc.Graph( figure=plotfig,
                          id="plot_fig",
                          style={"height": "350px"},
                          config={'displayModeBar': False}
),
                #dcc.Graph( figure=trend,
                #          id='trend 1',
                #          style={"height": "300px"},)
                #html.Div([html.Img(id='plot',src='data:image/png;base64,{}'.format(w1_encoded_image.decode()),
                 #                  style={'height':'300px', 'width':'550px'})]),
]
                                )
                            ],
                        )
                    ],
                ),
                dbc.Col(
                    width=7,
                    children=[
                        dbc.Card(
                            body=True,
                                style={"height": "530px","background": "#7FDBFF"},
                            children=[
                                dbc.FormGroup(
                                    [
                                        dcc.Loading(id="loading-2",color='white', type="default",
                                                    children=[
                                                        html.Div(
                                                            [dcc.Graph( figure=base_map,id="map-fig",style={"height": "500px"},config={'displayModeBar': False})
                                                                #html.Div(id="output",style={'font-weight': 'bold'})
                                                             ]
                                                                )
                                                             ]
                                                    )
                                    ]
                                             )
                                     ],
                                ),
                        html.Br(),
                        html.H5("Status of Critical Infrastructure", style={'textAlign': 'center',}),

                    cards
                    ],
                ),
            ]
        ),
    ],
)




@app.callback(
    [Output("Agent1", "color"),Output("Agent2", "color"),Output("Agent3", "color"),dash.dependencies.Output('table', 'data'),Output('table', 'columns'),
     Output(component_id='map-fig', component_property='figure'),Output('plot_fig','figure'),Output('plot_label','children')],
    [dash.dependencies.Input('drop_input', 'value'),Input("button-run", "n_clicks")])
def display_table(dropdown_value,b_click):
    time.sleep(1)
    if dropdown_value == 'Ward 1':
        return h_status(slope1),findstatus(ci_status[ci_status['Ward']==1]['Transport']),findstatus(ci_status[ci_status['Ward']==1]['Power']),w1_data.to_dict('records'),[{"name": i, "id": i} for i in w1_data.columns],w1_map,trend1,'Covid trend prediction in Ward-1 by ST-LSTM model'
    elif dropdown_value == 'Ward 2':
        return h_status(slope2),findstatus(ci_status[ci_status['Ward']==2]['Transport']),findstatus(ci_status[ci_status['Ward']==2]['Power']),w2_data.to_dict('records'),[{"name": i, "id": i} for i in w2_data.columns],w2_map,trend2,'Covid trend prediction in Ward-2 by ST-LSTM model'
    elif dropdown_value == 'Ward 3':
        return h_status(slope3),findstatus(ci_status[ci_status['Ward']==3]['Transport']),findstatus(ci_status[ci_status['Ward']==3]['Power']),w3_data.to_dict('records'),[{"name": i, "id": i} for i in w3_data.columns],w3_map,trend3,'Covid trend prediction in Ward-3 by ST-LSTM model'
    elif b_click==0:
        return 'secondary','secondary','secondary',grouped.to_dict('records'),[{"name": i, "id": i} for i in grouped.columns],base_map,plotfig, label_3d
    else:
        return result[4],result[6],result[5],grouped.to_dict('records'),[{"name": i, "id": i} for i in grouped.columns],fig_heat,maddpg_fig, "MADDPG Model Rewards"

        
    
@app.callback(
    [Output("button-run","disabled"),Output("action1","children"),Output("action2","children"),Output("action3","children"),Output('output','children')],
    [Input("button-run", "n_clicks")]
)
def summarize(n_clicks):
    #t0 = time.time()
    #t1 = time.time()
    #time_taken = f"Processed in {t1-t0:.2f}s"
    time.sleep(2)
    if n_clicks == 0:
        return False,"Run simulation to generate actions for Hospital","Run simulation to generate actions for Transportation CI","Run simulation to generate actions for Electric CI"," output will appear here, after Simulation..."
    else:
        return True,result[1],result[3],result[2],result[0]


if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)
