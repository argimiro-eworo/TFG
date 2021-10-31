#!/usr/bin/env python
# coding: utf-8

# # End to End Machine Learning Project
#  
# 
# ## Objetivo
# 
# La finalidad es, mediante un ejemplo práctico, conocer las principales pautas y procesos para la creación de un proyecto reproducible en machine learning. En este caso concreto, vamos a predecir los precios de las casas en la costa de california a través de unos datos y/o parámetros dados.
# 
# ## Metodología
# 
# Generalmente, procederemos con los siguientes pasos:  
# * Análisis y visualización de datos.  
# * Preparación de los datos para los algoritmos.  
# * Selección y entrenamiento de un modelo.  
# * Afinar el modelo.  
# * Lanzar el modelo.
# 
# ## Trabajos futuros
# 
# En un futuro se podrá ajustar mucho más el modelo, reduciendo el margen de error, usar algoritmos más precisos y simpificados.
# 
# ## Resultados
# 
# El resultado será un modelo capaz de predecir el valor del objeto en cuestión, que se podrá presentar al usuario como mejor convenga.
# 
# ## Recomendaciones
# 
# Es recomendable conocer muy bien los datos de los que se dispone así como los parámetros, para saber a qué modelo se ajustará mejor.

# # Herramientas
# 
# ## Importación de librerías
# Importamos las librerias necesarias para python

# In[1]:


import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# ## Importación de librerías locales
# Importamos todas las librerías locales necesarias

# In[2]:


import sys


# # Definición de parámetros
# Definimos los parámetros más importantes y/o necesarios para nuestro proyecto.

# In[3]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# 
# # Importación de datos
# Importamos los datos con los que vamos a trabajar.

# In[4]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# # Procesamiento de los datos
# 

# ## Estructura de datos

# **Vemos las primeras cinco columnas:**

# In[5]:


fetch_housing_data()
housing = load_housing_data()
housing.head()


# **Vemos una breve descripción de los datos, en particular el número de filas, tipos de atributos y el número de valores no nulos:**

# In[6]:


housing.info()


# **Para el atributo _ocean_proximity_ podemos ver las distintas categorías y el número de distridos que corresponden a cada:**

# In[7]:


housing["ocean_proximity"].value_counts()


# **El método _describe()_ nos enseña un resumen de los atributos numéricos:**

# In[8]:


housing.describe()


# **Otra forma de visualizar los datos es con histogramas:**

# In[9]:


housing.hist(bins=50, figsize=(20,15))
plt.show()


# **CREAMOS UN TEST AND SET**

# In[10]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[11]:


housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[12]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[13]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[15]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## Visualización de los datos

# Hasta ahora sólo hemos visto los datos por encima, ahora profundizaremos más.

# **VISUALIZACIÓN GRÁFICA**

# Ya que tenemos datos geográficos, es buena idea visualizarlos en un gráfico de dispersión, la densidad de población de todos los distritos:

# In[16]:


housing = strat_train_set.copy()


# In[17]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[18]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# A continuación veremos la variación de precios en cada área:

# In[19]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# Como se puede ver, los precios son más altos en zonas más cercanas al mar.

# ## Buscando correlaciones

# In[20]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# La correlación va de -1 a 1: cuanto más cerca del 1 más fuerte es la correlación y cuanto más al -1, es menos la correlación. Si es 0, es porque la correlación es nula.

# El mejor atributo para predecir el valor medio de una vivienda, es salario medio, los cuales tiene una fuerte correlación:

# In[21]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)


# In[22]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# Otra práctica aconsejable es experimentar con combinación de atributos.

# ## Preparación de los datos

# Es momento de preparar los datos para nuestros algoritmos de Machine Learning.
# Lo haremos con funciones para que sea reproducible y replicable.

# **LIMPIEZA DE DATOS**
# 
# Alguno algoritmos de ML no pueden funcionar cuando haya algunos valores que faltan. En nuestro caso  
# tenemos el atributo _total_bedrooms_. Tenemos tres opciones:  
# 1. Borrar los correspondientes distritos.  
# 2. Borrar el atributo.  
# 3. Rellenar los valores con algún otro valor (cero, media, mediana, etc).
# 
# En nuestro caso elegiremos a opción 1:

# In[23]:


housing.dropna(subset=["total_bedrooms"])


# In[24]:


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)


# **ATRIBUTOS DE TEXTO Y CATEGÓRICOS**
# 
# Ya hemos tratado los atributos numéricos, ahora lo haremos con los de texto. En nuestro caso tenemos uno _ocean_proximity_.  
# Para los diez primeros valores, tenemos:

# In[25]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# Muchos algoritmos de ML prefieren trabajar antes con números, así que convertiremos nuestro atributo textual a numérico: 

# In[26]:


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_


# Creamos un atributo binario por categoría:

# In[27]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()
cat_encoder.categories_


# **TRANSFORMACIONES PERSONALIZADAS**
# 
# Una buena práctica es crear operaciones personalizadas de limpieza de datos o combinación de atributos.  
# Se puede hacer con una clase que contiene los métodos que queramos. En nuestro caso:

# In[28]:


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# **PIPELINES**
# 
# Tenemos una transformación de datos que es necesario ejecutar en el orden correcto. Para tal fin,  
# tenemos la clase _Pipeline_ de Scikit-Learn:

# In[29]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# Sería conveniente tener un transformador que maneje todas las columnas aplicando las transformaciones apropiadas  
# para cada una. Para eso tenemos _ColumnTransformer_ de **Scikit-Learn**:

# In[30]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# ## SELECCIÓN Y ENTRENAMIENTO DE UN MODELO

# ### MODELO DE REGRESIÓN LINEAL

# In[31]:


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#A continuación lo probamos con nuestros datos
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
lin_rmse


# Funciona, aunque la predicción no sea exacta. No es la mejor, pero mejor algo que nada.

# ### MODELO DE ÁRBOLES DE REGRESIÓN

# In[32]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ¡¡Sin error!!

# Ahora probamos otra vez el modelo con validación cruzada. Dividimos los datos en 10 partes y evaluamos el modelo  
# en 10 veces.

# In[33]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)


# El modelo ya no parece tan bueno, de hecho es peor que el de regresión lineal.  
# Vamos a lanzar dicho moledo con los mismos datos partidos:

# In[34]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# El modelo de árboles de regresión tiene sobreajuste, lo que le hace peor que el de regresión lineal.

# ### MODELO _RandomForestRegressor_

# El modelo funciona entrenando varios árboles de decisión en varios subconjuntos aleatorios y luego haciendo  
# la media de sus predicciones.

# In[35]:


""" forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores) """


# Este modelo es bastante mejor. Es modelo también presenta sobreajuste, una solución sería meter más datos

# ## AJUSTE DEL MODELO ELEGIDO

# ### BÚSQUEDA EN CUADRÍCULA

# Usaremos la función _GridSearchCV_ de Scikit-Learn’s para encontrar la mejor combinación de hyperparámetros  
# de nuestro modelo. La búsqueda se hará en 18 combinaciones de los hyperparámetros del modelo _RandomForestRegressor_ y entrenará cada modelo 5 veces. En total tendremos 90 rondas de entrenamiento:

# In[36]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)


# In[37]:


grid_search.best_params_ #Mejor combinación de parámetros


# In[38]:


grid_search.best_estimator_ #Mejor estimador


# In[39]:


cvres = grid_search.cv_results_ #Resultado de las evaluaciones
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ### BÚSQUEDA ALEATORIA

# La búsqueda en cuadrícula es buena cuando estas probando algunas combinaciones, pero cuando el número de hyperparámetros a analizar es largo es preferible usar la búsqueda aleatoria _RandomizedSearchCV_. Hace lo mismo que la anterior, pero este evalúa un número dado de combinaciones aleatorias selección un valor aleatorio para cada hyperparámetro en cada iteración.

# ## EVALUACIÓN DEL SISTEMA

# Después de analizar y ajustar varios modelos, toca evaluar el modelo final en _test set_:

# In[40]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evalúa a 47,730.2


# ## LANZAR, MONITORIZAR Y MANTENER EL SISTEMA

# Ya tenemos nuestro modelo listo para ser lanzado y usado. Una manera sencilla de hacerlo es a través de una web, donde un usuario podrá seleccionar el distrito deseado y darle al botón de estimar precio. Con eso, internamente, se llama a la función _predict()_ creada previamente y que lanza las operaciones de predicción de nuestro modelo y devuelve la solución que a su vez se presenta al usuario.

# ![](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781492032632/files/assets/mls2_0217.png)

# Otra opción es desplegar el modelo en la nube, como por ejemplo en el  _Google Cloud AI Platform_. Simplemente hay que guardar el modelo usando _joblib_ y subirlo a _Google Cloud Storage (GCS)_ luego ir a la plataforma de google cloud y crear una nueva versión del modelo que apunta a un archivo GCS.

# El despliegue no lo es todo, también es necesario crear un código de monitoreo de nuestro modelo para posibles fallos y alertas, ya sea por una componente desactualizada o una que falte. El sistema de monitoreo deberá tener unas pautas de actuación dependiente del tipo de fallo.

# # References

# [1] Aurélien Géron. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition. 2019. [URL](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch02.html#idm45022192813432)

# In[ ]:




