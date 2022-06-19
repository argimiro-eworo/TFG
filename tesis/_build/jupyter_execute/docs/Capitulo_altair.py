#!/usr/bin/env python
# coding: utf-8

# # El Tiempo En Nueva York

# Primero, cargamos los datos brutos con los que vamos a trabajar:

# In[1]:


import pandas as pd
import altair as alt
data = pd.read_csv('https://raw.githubusercontent.com/toddwschneider/nyc-citibike-data/master/data/daily_citi_bike_trip_counts_and_weather.csv')
data.head()


# Esos datos se cargan desde una web y se guardan como un **DataFrame** de **Pandas** y desde ahí ya podemos explorarlos con **Altair**.

# Empezamos por las precipitaciones, usamos _tick marks_ para ver la distribución de sus valores:

# In[2]:


alt.Chart(data).mark_tick().encode(
    x='precipitation',
)


# Vemos que para valores bajos, las precipitaciones están sesgadas. Es decir, cuando llueve no llueve mucho. 
# 
# Para visualizar mejor los datos, creamos un histograma con los datos de precipitación. Para ellos primero discretizamos los valores y luego codificamos:

# In[3]:


alt.Chart(data).mark_bar().encode(
    alt.X('precipitation', bin=True),
    y='count()'
)


# A continuación, veamos como cambia el tiempo en Nueva York a lo largo del año:

# In[4]:


alt.Chart(data).mark_line().encode(
    x='month(date):T',
    y='average(precipitation)'
)


# Es esta gráfica podemos ver que, de media, las precipitaciones son mayores en invierno que en verano

# Viendo los datos de temperatura y precipitación, igual nos gustaría ver el agregado por año y mes, en vez de sólo el mes. Esto es útil para ver tendencias estacionales. Nos gustaría también ver, las temperatuas máxima y mínima para cada mes:

# In[5]:


alt.Chart(data).mark_line().encode(
    x='yearmonth(date):T',
    y='max(max_temperature)',
)


# Con este gráfico, vemos que la temperatura máxima bajó en 2014. Para verlo mejor, hacemos la media de las temperaturas diarias para cada año:

# In[6]:


alt.Chart(data).mark_line().encode(
    x='year(date):T',
    y='mean(max_temperature)',
)


# Para verlo más claro, usaremos un diagrama de barras horizontales ordenado por años:

# In[7]:


alt.Chart(data).mark_bar().encode(
    x='mean(max_temperature)',
    y='year(date):O'
)


# Veamos también cómo cambia el rango de temperaturas a lo largo del año:

# In[8]:


alt.Chart(data).mark_bar().encode(
    x='mean(temp_range):Q',
    y='year(date):O'
).transform_calculate(
    temp_range="datum.max_temperature - datum.min_temperature"
)


# A continuación, vamos a explorar los meses.
# 
# Primero creamos un diagrama de barras con la cantidad de cada mes:

# In[9]:


alt.Chart(data).mark_bar().encode(
    x='month(date):N',
    y='count()',
    color='month',
)


# Lo siguiente será personalizar los colores con los que representamos cada mes. Los meses más frescos los representamos con colores suaves y los calurosos con colores más fuertes:

# In[10]:


scale = alt.Scale(domain=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                  range=['#F2D7D5', '#FADBD8', '#E8DAEF', '#F5B7B1 ', '#F5CBA7', '#E59866', '#E67E22', '#B03A2E', '#BA4A00', '#2980B9','#138D75', '#1ABC9C'])


# In[11]:


alt.Chart(data).mark_bar().encode(
    x=alt.X('month(date):N', title='Month of the year'),
    y='count()',
    color=alt.Color('month', legend=alt.Legend(title='Weather type'), scale=scale),
)


# Combinando lo anteriormente expuesto, tenemos un gráfico que relaciona temperatura, mes y precipitaciones:

# In[12]:


alt.Chart(data).mark_point().encode(
    alt.X('max_temperature', title='Maximum Daily Temperature (C)'),
    alt.Y('temp_range:Q', title='Daily Temperature Range (C)'),
    alt.Color('month', scale=scale),
    alt.Size('precipitation', scale=alt.Scale(range=[1, 200])),
).transform_calculate(
    "temp_range", "datum.max_temperature - datum.min_temperature"
).properties(
    width=600,
    height=400
).interactive()


# Construimos un histograma de los meses:

# In[13]:


alt.Chart(data).mark_bar().encode(
    x='count()',
    y='month:N',
    color=alt.Color('month', scale=scale),
)


# finalmente, concatenamos dicho histograma con el gráfico anterior y construimos otro gráfico interactivo, en el que seleccionas el rango de temperaturas deseadas y el histograma refleja el contenido seleccionado:

# In[14]:


brush = alt.selection(type='interval')

points = alt.Chart().mark_point().encode(
    alt.X('max_temperature:Q', title='Maximum Daily Temperature (F)'),
    alt.Y('temp_range:Q', title='Daily Temperature Range (F)'),
    color=alt.condition(brush, 'month', alt.value('lightgray'), scale=scale),
    size=alt.Size('precipitation:Q', scale=alt.Scale(range=[1, 200]))
).transform_calculate(
    "temp_range", "datum.max_temperature - datum.min_temperature"
).properties(
    width=600,
    height=400
).add_selection(
    brush
)

bars = alt.Chart().mark_bar().encode(
    x='count()',
    y='month:N',
    color=alt.Color('month', scale=scale),
).transform_calculate(
    "temp_range", "datum.max_temperature - datum.min_temperature"
).transform_filter(
    brush
).properties(
    width=600
)

alt.vconcat(points, bars, data=data)


# In[ ]:




