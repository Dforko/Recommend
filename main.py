from typing import Union
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


app = FastAPI()

db = pd.read_parquet('dbSm.parquet', engine='pyarrow')

@app.get("/")
def read_root():
    return {"Moviex": "Movie Recommendation System: What do you want to watch today?"}



@app.get('/get_actor/{nombre_actor}')
def get_actor( nombre_actor ):
    actor_info = db[(db['cast'].str.contains(nombre_actor))]
    actor_cant_movies=actor_info['cast'].shape[0]
    actor_return =round(actor_info['return'].sum(),2)
    if  nombre_actor not in actor_info.values:
        return(f"{nombre_actor} no es un nombre válido. Chequea la ortografía y mayúsculas.")
    return {'El actor/la actriz': nombre_actor, 
            'ha participado de ':actor_cant_movies,
            'conseguiendo un retorno de': actor_return,
            'con un promedio por filmación de':round(actor_return/actor_cant_movies,2)}

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
    if Dia not in days:
        return f"{Dia} no es un día válido."
    return {
        'La cantidad de': respuesta,
        'películas fueron estrenadas en los días': Dia    }



@app.get('/cantidad_filmaciones_mes/{Mes}')
def cantidad_filmaciones(Mes:str):
    m_dic = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    nro_mes = m_dic.get(Mes)
    if nro_mes is None:
        return {"error": f"{Mes} is not a valid month."}
    count = sum(1 for date in db['release_date'] if date.month == nro_mes)
    return {
        'Fueron estrenadas la cantidad de': count,
        'películas en': Mes
    }
    
    
@app.get('/votos_titulo/{titulo_de_la_filmacion}')
def votos_titulo(titulo_de_la_filmacion: str):
    movie_info = db[db['title'] == titulo_de_la_filmacion]
    if titulo_de_la_filmacion not in movie_info.values:
        return f"{titulo_de_la_filmacion} no es un título válido. Chequea que el título sea el correcto, la ortografía y mayúsculas."
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
    if titulo_de_la_filmacion not in movie_info.values:
        return f"{titulo_de_la_filmacion} no es un título válido. Chequea que el título sea el correcto, la ortografía y mayúsculas."
    return {
        'La película':titulo_de_la_filmacion,
        'fue estrenada en el año': str(movie_info.release_year.values[0]),
        'con un score/popularidad de': str(movie_info.popularity.values[0])
    }

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director):
    director_db = db[(db['director'] == nombre_director)]
    suma_retorno = director_db['return'].sum()
    # Crear un DataFrame con las columnas especificadas
    resultado = director_db[['title', 'release_year', 'return', 'budget', 'revenue']]
    # Renombrar las columnas del DataFrame
    resultado = resultado.rename(columns={'title': 'Título', 'release_year': 'Año', 'return': 'Retorno', 'budget': 'Costo', 'revenue': 'Ganancia'})
    # Ordenar el DataFrame por la columna 'Año' en orden ascendente
    resultado = resultado.sort_values(by='Año', ascending=True)
    # Change the data type of specific columns to int
    resultado[['release_year', 'budget', 'revenue']] = resultado[['release_year', 'budget', 'revenue']].astype(int)
    # Devolver la suma de retorno y el DataFrame resultado
    return {'El retorno total de': nombre_director,
            'es de' : round(suma_retorno, 2),
            'Estas son sus películas, año de lanzamiento, retorno individual, costo y ganancia:': resultado}

# RECOMMENDATION FUNCTION 

db = dbSm
dbSm['description'] = dbSm['title'] + dbSm['overview'] + dbSm['tagline']

cv = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = cv.fit_transform(dbSm['description'])

nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(count_matrix)


smd = dbSm.reset_index()
titles = smd['title']

indices = pd.Series(smd.index, index=smd['title'])


@app.get('/get_recommendations/{title}')
def get_recommendations(title:str):
       
    # Verificamos si el titulo ingresado se encuentra en el df
    if title not in dbSm['title'].values:
        return 'La pelicula no se encuentra en el conjunto de la base de datos.'
    else:
        # Obtenemos el índice de la película que coincide con el título
        index = indices[dbSm.title]

        # Obtenemos las puntuaciones de similitud de las 5 peliculas más cercanas
        distances, indices_knn = nn.kneighbors(count_matrix[index], n_neighbors=6)  

        # Obtenemos los indices de las peliculas
        movie_indices = indices_knn[0][1:]  

        # Devolvemos las 5 peliculas mas similares
        return {'lista recomendada': dbSm['title'].iloc[movie_indices].tolist()}
