#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pandas as pd
import seaborn as sns
from siuba import *
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# # Objective 1

# In[3]:


df = pd.read_csv('data/MVP01-10_Bus_by_Pax.csv',sep=',')


#drop duplicates
df.drop_duplicates()


df.info()


# In[4]:


print(df)


# In[5]:


df1=(df
  >> group_by(_.capacity)
  >> summarize(number = _.number.sum())
  >>separate(
    _.capacity,
    ['lower','higher']
    , sep='-'
    )
  )
#replace values
df1.iat[12,2]=9
df1.iat[13,2]=71

df1['higher'] = df1['higher'].astype(int)

df1=(df1
    >>mutate(total=_.higher*_.number))

sort_df=df1.sort_values(by=['total'],ascending=False)
sort_df=sort_df.reset_index(drop=True)
print(sort_df)


# In[6]:


print(df['capacity'].unique())


# In[7]:


top5=sort_df[:5]
print(top5)

test=top5['number']


# In[8]:


df_new=(
    df
    >>filter(_.capacity.isin(
    [' 10-15','> 70','41-45','46-50','21-25']))
)




x=df['year'].unique()
p=['ten','seventy','fortyone']


print(df_new)


# In[9]:


##  Code for Bubble Chart animation
fig = px.scatter(data_frame = df_new, x = 'year', y = 'number', size = 'number', 
                 color = 'capacity', hover_name = 'capacity', log_x = True,
                    size_max = 50, 
                    range_x = [2005,2021],
                    range_y = [0,7000],

                ## these 2 parameters style the animation
                    animation_frame = 'year',
                    animation_group = 'capacity'
                    
                ## display text
                ,text = 'capacity'
                
                ## display Title
                , title = 'Growth in number of buses'

                    )

                    
fig.update_xaxes(title_text = 'Year', title_font=dict(size=20))
fig.update_yaxes(title_text = 'Number of buses', title_font=dict(size=20))
fig.update_layout(title_font_size = 25, title_font_color='crimson',title_x=0.5)



fig.show()


# In[157]:


df2 = pd.read_csv('data/public-transport-utilisation-average-public-transport-ridership.csv', sep=',')
df2.head(5)
x=df2['year'].unique()


mrt=df2[df2['type_of_public_transport']=='MRT']['average_ridership']
lrt=df2[df2['type_of_public_transport']=='LRT']['average_ridership']
bus=df2[df2['type_of_public_transport']=='Bus']['average_ridership']
taxi=df2[df2['type_of_public_transport']=='Taxi']['average_ridership']

plt.stackplot(x,lrt,taxi,mrt,bus,labels=['lrt','taxi','mrt','bus'],alpha=0.8)
plt.title("Average ridership for public transport",fontsize=16,weight='bold')
plt.xlabel ('Year',fontsize=16)
plt.ylabel ('Average ridership',fontsize=16)
plt.legend(loc='upper left')
plt.show()


# In[ ]:





# In[14]:


# 
print(df2['average_ridership'].min())
print(df2['average_ridership'].max())


# In[15]:


import matplotlib.ticker as ticker

fig, ax = plt.subplots()
sns.violinplot(data=df2,x='type_of_public_transport',y='average_ridership',ax=ax,spanmode='hard')
ax.yaxis.set_major_locator(ticker.MultipleLocator(1000000))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_ylim(0,4500000)
ax.set_xlabel("Transport", fontsize=15)
ax.set_ylabel("Average ridership", fontsize=15)
ax.set_title('Average public transport ridership', fontsize=20,fontweight='bold')
plt.show()


# # Objective 2

# In[16]:


df = pd.read_csv('data/bus_services.csv', sep=',')
df.head(5)

#Remove the unecessary columns 
df.drop(['Unnamed: 0', 'Direction','OriginCode', 'DestinationCode', 'LoopDesc'],inplace=True,axis=1)

#Find the number of duplicates
df.drop_duplicates()


from siuba import *


# In[17]:


#Handle missing values
df.replace(to_replace='^-*$', value=np.nan, regex=True, inplace=True)

#Fill the missing values
df=df.dropna()
df=df.reset_index(drop=True)


# In[18]:


df.isnull().sum()
df=df.melt(id_vars=['ServiceNo', 'Operator','Category'], var_name='Subject', value_name='Frequency')


print(df)


# In[19]:


# df>>mutate()


# In[20]:


df[['Minimum', 'Maximum']] = df['Frequency'].str.split('-', 1, expand=True)
df = df.replace(to_replace='None', value=np.nan).dropna()

print(df.head())
print(type(df.Minimum))


# In[22]:


df["Median"] = df[["Minimum", "Maximum"]].median(axis=1)


# In[23]:


print(df)


# In[24]:


df['Minimum'] = df['Minimum'].astype(int)
df['Maximum'] = df['Maximum'].astype(int)
df['Median'] = df['Median'].astype(int)


df.dtypes
#  (f) comvert from wide to long
df1 = pd.melt(df, id_vars=['ServiceNo','Operator','Category','Subject','Frequency'], var_name='Range',value_name='Time')
df1['Time'] = df1['Time'].astype(int)


# In[26]:


print(df1)


# In[27]:


import plotly.graph_objects as go


fig = go.Figure()

####For AM PEAK FREQ
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='AM_Peak_Freq')&(df1['Operator']=="SMRT")]['Time']
                          ,marker=dict(
                             color= 'rgb(255,178,102)'
                             , opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='SMRT'
                          ))
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='AM_Peak_Freq')&(df1['Operator']=="GAS")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='GAS'
                          ))
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='AM_Peak_Freq')&(df1['Operator']=="SBST")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='SBST'
                          ))

fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='AM_Peak_Freq')&(df1['Operator']=="TTS")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='TTS'
                          ))


###FOR PM PEAK FREQ
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='PM_Peak_Freq')&(df1['Operator']=="SMRT")]['Time']
                          ,marker=dict(
                             color= 'rgb(255,178,102)'
                             , opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='SMRT'
                          ))
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='PM_Peak_Freq')&(df1['Operator']=="GAS")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='GAS'
                          ))
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='PM_Peak_Freq')&(df1['Operator']=="SBST")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='SBST'
                          ))

fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='PM_Peak_Freq')&(df1['Operator']=="TTS")]['Time']
                          ,marker=dict(
                             opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='TTS'
                          ))

#FOR OFFpeak PM
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='AM_Offpeak_Freq')]['Time']
                          ,marker=dict(
                             color= 'rgb(238,130,238)'
                             , opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='AM_Offpeak_Freq'
                          ))

#FOR OFFpeak AM
fig.add_trace(go.Histogram(x=df1[(df1['Subject']=='PM_Offpeak_Freq')]['Time']
                          ,marker=dict(
                             color= 'rgb(64,224,208)'
                             , opacity = 0.82
                             # ,line = dict(color='DarkSlateGrey', width=1)
                             )
                           , xbins=dict(
                               size=2   ## (optional)
                              ),
                           name='PM_Offpeak_Freq'
                          ))
      

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="AM_Peak_Freq",
                     method="update",
                     args=[{"visible": [True, True,True,True,False,False, False, False,False,False]},
                           {"title": "Morning Peak Frequency",
                            "annotations": []}]),
                dict(label="PM_Peak_Freq",
                     method="update",
                     args=[{"visible": [False,False,False,False,True,True, True,True,False,False]},
                           {"title": "Evening Peak Frequency",
                            "annotations": []}]),
                dict(label="AM_Offpeak_Freq",
                     method="update",
                     args=[{"visible": [False,False,False,False,False,False,False,False,True,False]},
                           {"title": "Morning Offpeak Frequency",
                            "annotations": []}]),
                dict(label="PM_Offpeak_Freq",
                     method="update",
                     args=[{"visible": [False,False,False,False,False,False,False,False,False,True]},
                           {"title": "Evening Offpeak Frequency",
                            "annotations": []}]),
            ]),
        )
    ])


fig.update_xaxes(title_text = 'Timing(in minutes)', title_font=dict(size=20))
fig.update_yaxes(title_text = 'Count', title_font=dict(size=20))
fig.update_layout(title = 'Frequency of buses',title_font_size = 20, title_font_color='crimson',barmode='stack')
fig.update_layout(template='seaborn')
fig.show()


# In[126]:


df1 = pd.read_csv('data/statistic_id1007813_number-of-major-delays-lasting-more-than-30-minutes-of-mrt-singapore-2015-2021.csv', sep=',')
df2 = pd.read_csv('data/statistic_id1007841_number-of-major-delays-lasting-more-than-30-minutes-of-lrt-singapore-2015-2021.csv', sep=',')


# to append df2 at the end of df1 dataframe
df3=pd.concat([df1,df2],axis=1)

#Rename the columns
df3.rename(columns = {'Number of delays lasting more than 30 minutes of the mass rapid transit (MRT) network in Singapore from 2015 to 2021': 'Year', 'Unnamed: 1':'Number of delays','Number of delays lasting more than 30 minutes of light rail transit (LRT) network in Singapore from 2015 to 2021':'Year2'},inplace=True)
df3.pop('Year2')

df3.columns=['Year','MRT','LRT']

print(df3)


# In[127]:


test = df3.melt(id_vars="Year", var_name=["Transport"], value_name="Number of Delays")
print(test.info())

df3['Year'] = df3['Year'].astype(int)


# In[129]:


df3


# In[149]:


sns.set_style("whitegrid")
fig,ax=plt.subplots(nrows=1,
                  ncols=2,
                  sharey=True, #must share the same scale for the y axis
                  figsize=(7,4))

fig.suptitle('Number of delays in MRT and LRT',fontsize=20,color='black',weight='bold')

ax[0].set_ylabel('Number of Delays',fontsize=15)
sns.lineplot(data=df3,x='Year',y='MRT',color='r',ax=ax[0])
ax[0].set(ylim=(0, 20))
plt.yticks(np.arange(0, 20, 2))
sns.lineplot(data=df3,x='Year',y='LRT', color="b", ax=ax[1])
fig.legend(labels=["MRT","LRT"])


# In[ ]:


sns.set()

g = sns.FacetGrid(test, col="Transport", height=4, aspect=0.8)

g = g.map(sns.lineplot, "Year", "Number of Delays",color='r')

plt.subplots_adjust(wspace=0.2)
g.set(xlim=(2015, 2020))
g.set_ylabels("Number of Delays", fontsize=15)
g.set_xlabels("Year", fontsize=15)


# # https://www.singstat.gov.sg/publications/cop2010/census10_stat_release3

# In[159]:


df5= pd.read_excel('data/travelling-time-to-work2010.xlsx', na_values=[' '])
df6= pd.read_excel('data/time-taken-to-travel-2020.xlsx', na_values=[' '])


# In[160]:


#delete the columns
df5.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 5', 'Unnamed: 6','Unnamed: 8', 'Unnamed: 9'],axis=1,inplace=True)
df6.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 5', 'Unnamed: 6','Unnamed: 8', 'Unnamed: 9'],axis=1,inplace=True)

# dropping the rows having NaN values
df5=df5.dropna()
df6 = df6.dropna()
df6['Year']=2020
df5['Year']=2010


# In[161]:


print(df5)
print(df6)


# In[45]:


df6.info()


# In[93]:


#add a column year
df6['Year']=2020
df5['Year']=2010

time_df = pd.concat([df5, df6])
time_df=time_df.reset_index(drop=True)
time_df.rename(columns={'Mode of Transport': 'Mode', 'Up to 15 mins': '<15mins'}, inplace=True)
time_df['Mode'] = time_df['Mode'].replace({'MRT/LRT Only': 'MRT Only', 'Taxi/Private Hire Car Only': 'Taxi Only','MRT/LRT & Public Bus Only':'MRT & Public Bus Only'})


# In[94]:


time_df['Mode'].unique()


# In[95]:


time_df = (time_df
     >> filter(_.Mode.isin( ['Public Bus Only','MRT Only','MRT & Public Bus Only','Taxi Only']))
     )


# In[114]:


time_df= time_df.melt(id_vars=['Mode', 'Total', 'Year'], value_vars=['<15mins', '16 - 30 mins'], var_name='category', value_name='number')


# In[198]:


time_df.info()

time_df['Total'] = time_df['Total'].astype(int)
time_df['number'] = time_df['number'].astype(int)


# In[204]:



time_df.info()


# In[216]:


# Plot the grouped bar chart
g_bar = sns.catplot(x="Mode", y="Total", hue="Year", col="category", data=time_df, kind="bar", aspect=0.7)
g_bar.set_xticklabels(rotation=30)

g_bar.fig.suptitle("Travel timing in 2010 vs 2020")
plt.subplots_adjust(top=0.8)
# Show the plot
plt.show()

