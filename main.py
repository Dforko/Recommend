from typing import Union
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


import warnings; warnings.simplefilter('ignore')

# Creamos una instancia de la aplicación FastAPI

app = FastAPI(title='Moviex: Recomendación System- Machine Learning Operations',
            description='Diego Forcato')

# Especificamos los campos y sus tipos de datos correspondientes que se esperan en los objetos Item.

class Item(BaseModel):
    title: str
    genres: str
    overview: str
    tagline: str
    budget: int
    revenue: int

# Inicializamos la  variable global "db" antes de que la aplicación FastAPI se inicie y establecemos conexión con la base de datos.
db = None
@app.on_event('startup')
async def startup():
    global db
    
    
    db = pd.read_parquet('db10000.parquet', engine='pyarrow')

# Comenzamos a codear las funciones requeridas

@app.get("/")
def read_root():
    return {"Moviex": "Movie Recommendation System: What do you want to watch today?"}



# Endpoint para obtener la cantidad de películas estrenadas en un día ingresado.

@app.get('/cantidad_filmaciones_dia/{Dia}')
def cantidad_filmaciones_dia(Dia:str):
    # Creamos diccionario para normalizar
    days = {
    'Lunes': 'Monday',
    'Martes': 'Tuesday',
    'Miércoles': 'Wednesday',
    'Jueves': 'Thursday',
    'Viernes': 'Friday',
    'Sábado': 'Saturday',
    'Domingo': 'Sunday'}
    day = days[Dia]
    # Filtramos los duplicados del dataframe y calculamos
    lista_peliculas_day = db[db['release_date'].dt.day_name() == day]
    respuesta = lista_peliculas_day.shape[0]
     # Verificar si el valor del día es válido
    if Dia not in days:
        return f"{Dia} no es un día válido. Chequea la ortografía y mayúscula."
    return {
        'La cantidad de': respuesta,
        'películas fueron estrenadas en los días': Dia    }

# Endpoint para obtener la cantidad de películas estrenadas en el mes ingresado.

@app.get('/cantidad_filmaciones_mes/{Mes}')
def cantidad_filmaciones(Mes:str):
    # Creamos diccionario para obtener el mes en número
    m_dic = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    nro_mes = m_dic.get(Mes)
     # Verificamos si el valor del mes es válido
    if nro_mes is None:
        return {"error": f"{Mes} no es un mes válido. Chequea la ortografía y mayúscula."}
    count = sum(1 for date in db['release_date'] if date.month == nro_mes)
    return {
        'Fueron estrenadas la cantidad de': count,
        'películas en': Mes
    }
    
    
@app.get('/votos_titulo/{titulo_de_la_filmacion}')
def votos_titulo(titulo_de_la_filmacion: str):
    movie_info = db[db['title'] == titulo_de_la_filmacion]
     # Verificamos si el título de la película es válido
    if titulo_de_la_filmacion not in movie_info.values:
        return f"{titulo_de_la_filmacion} no es un título válido. Chequea que el título sea el correcto, la ortografía y mayúsculas."
   # Verificamos que la pelicula ingresada posea suficientes valoraciones
    if movie_info.vote_count.values[0] < 2000:
        return f"{titulo_de_la_filmacion} no posee suficientes valoraciones."
    else:
        return {
            'La película': str(movie_info.title.values[0]),
            'fue estrenada en el año': str(movie_info.release_year.values[0]),
            'La misma cuenta con una valoración total de': str(movie_info.vote_count.values[0]),
            'con un promedio de': str(movie_info.vote_average.values[0])
        }
        
@app.get('/score_titulo/{titulo_de_la_filmacion}')
def score_titulo(titulo_de_la_filmacion: str):
    movie_info = db[db['title'] == titulo_de_la_filmacion]
    # Verificamos si el título de la película es válido
    if titulo_de_la_filmacion not in movie_info.values:
        return f"{titulo_de_la_filmacion} no es un título válido. Chequea que el título sea el correcto, la ortografía y mayúsculas."
    return {
        'La película':titulo_de_la_filmacion,
        'fue estrenada en el año': str(movie_info.release_year.values[0]),
        'con un score/popularidad de': str(movie_info.popularity.values[0])
    }
    
@app.get('/get_actor/{nombre_actor}')
def get_actor( nombre_actor ):
    actor_info = db[(db['cast'].str.contains(nombre_actor))]
    actor_cant_movies=actor_info['cast'].shape[0]
    actor_return =round(actor_info['return'].sum(),2)
    # Verificamos si el nombre del actor/la actriz es válido
    if  nombre_actor not in actor_info.values:
        return(f"{nombre_actor} no es un nombre válido. Chequea la ortografía y mayúsculas.")
    return {'El actor/la actriz': nombre_actor, 
            'ha participado de ':actor_cant_movies,
            'conseguiendo un retorno de': actor_return,
            'con un promedio por filmación de':round(actor_return/actor_cant_movies,2)}
    
@app.get('/get_director/{nombre_director}')
def get_director( nombre_director ):
    director_db = db[(db['director'].str.contains(nombre_director))]
    resultado = director_db[['title', 'release_year', 'return', 'budget', 'revenue']]
    director_db[['release_year', 'budget', 'revenue']] = director_db[['release_year', 'budget', 'revenue']].astype(int)
    suma_retorno =round(director_db['return'].sum(),2)
    resultado = resultado.sort_values(by='release_year', ascending=True)
    # Cambiamos los nombres de las columnas a español 
    resultado = resultado.rename(columns={'title': 'Título', 'release_year': 'Año', 'return': 'Retorno', 'budget': 'Costo', 'revenue': 'Ganancia'})
     # Verificamos si el nombre del director/a es válido
    if  nombre_director not in director_db.values:
        return(f"{nombre_director} no es un nombre válido. Chequea la ortografía y mayúsculas.")
    return {'El retorno total de': nombre_director,
            'es de' : round(suma_retorno, 2),
            'Estas son sus películas, año de lanzamiento, retorno individual, costo y ganancia:': resultado.astype(str)}



# RECOMMENDATION FUNCTION 



# Funcion Machine Learning - "Modelo de Vecinos mas Cercanos" 
# Devuelve las 5 películas recomendadas basadas en la película que ingresamos como input .
@app.get("/get_recommendation/{movie_title}")
def get_recommendation(movie_title):

    # Convertimos el título de la película a minúsculas
    movie_title = movie_title.lower()

    # Buscamos la película por título en la columna 'title'
    movie = db[db['title'].str.lower() == movie_title]

    if len(movie) == 0:
        return f"{movie_title} no es un nombre válido. Chequea la ortografía y mayúsculas."

    # Obtenemos  género y  popularidad de la película
    genre = movie['genres'].values[0]
    popularity = movie['popularity'].values[0]

    # Creamos una matriz de para el modelo K-NN
    features = db[['popularity']]
    genres = db['genres'].str.get_dummies(sep=' ')
    features = pd.concat([features, genres], axis=1)

    # Creamos el modelo modelo K-NN
    nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn_model.fit(features)

    # Devuelve las películas más similares
    _, indices = nn_model.kneighbors([[popularity] + [0] * len(genres.columns)], n_neighbors=6)

    # Retornamos los títulos de las películas recomendadas
    return db.iloc[indices[0][1:]]['title']


# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=800
