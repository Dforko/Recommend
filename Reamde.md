# Moviex: Recomendador de películas (Proyecto Individual 1)

# Autor: `Dr. Diego Forcato` 

## Tabla de Contenidos

- [Descripción](#descripción)
- [ETL (Limpieza de Datos) ](#Proceso)
- [Metadata](#diccionario)
- [Modelo de Machine Learning](#diccionario)





## Descripción

El objetivo de este proyecto individual es crear desde cero un modelo de recomendación de películas utilizando Machine Learning y desplegarlo en el mundo real a través de una API creada utilizando FastAPI y desplegada mediante el servicio Render para realizar consultas de películas on kine.

El ciclo de vida de un proyecto de Machine Learning contempla desde el tratamiento y recolección de los datos (Data Engineer) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.


La madurez del conjunto de datos `dataset_movies.csv` del cual partimos es baja: Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevas películas o series, por lo que se requiere un proceso de Limpieza de datos intensivo.




## Proceso de Limpieza de Datos

El proceso de limpieza de datos incluye en varios pasos:

**1-Carga de Datos**: Se realizará la carga de los datos a partir de dos archivos csv (`dataset_movies.csv` y `credits.csv`) en sendos DataFrames utilizando la biblioteca Pandas.

**2-Transformaciones**: 

**Eliminación de Columnas Innecesarias** : Eliminar las columnas que no serán utilizadas, tales como  **`video`**,**`imdb_id`**, **`adult`**, **`original_title`**, **`poster_path`** y **`homepage`**.

**Desanidación de campos**: Algunos campos, como **`belongs_to_collection`**, **`production_companies`** y otros (ver diccionario de datos) están anidados (tienen un diccionario o una lista como valores en cada fila), lo cual puede dificultar el accionar de la API. Se extrajeron los datos relevantes de estas columnas y se crearon nuevos  campos en el DataFrame

 **Limpieza de Datos Faltantes** : Se identificaron columnas con datos faltantes. Se aplicaron técnicas de imputación o eliminación de filas/columnas dependiendo del caso. Por ej. los valores nulos de los campos **`revenue`**, **`budget`**  se rellenaron con  el número **`0`**, mientras que los valores nulos del campo **`release date`** fueron eliminados.


**Normalización de Datos** : Se aplicaron tecnicas de transformación exigidas a las columnas, como por ejemplo convertir formatos de fechas (Debian tener el formato **`AAAA-mm-dd`**), se eliminaron caracteres especiales, etc. Además se crearon las columnas **`release_year`** extrayendo el año de la fecha de estreno a partir de la columna **`release_date`** y la columna llamada **`return`** (retorno de inversión), dividiendo las columnas **`revenue`** y **`budget`** (**`revenue / budget`**, cuando no hubo datos disponibles para calcularlo, se usó el valor **`0`**).

**Validación y Control de Calidad** : Se realizó la validación de los datos limpios para asegurar el cumplimiento de los requisitos de la API.





## Metadata (Diccionarios de datos)

Aqui se muestran los diccionario con la descripción de cada columna del archivo `dataset_movies.csv` y `credits.csv`:

### `dataset_movies.csv`

---                --------------  

`adult`:   Indica si la película tiene califiación X, exclusiva para adultos.<br>
`belongs_to_collection`:   Un diccionario que indica a que franquicia o serie de películas pertenece la película<br>	
`budget`:   El presupuesto de la película, en dólares<br>
`genres`:   Un diccionario que indica todos los géneros asociados a la película	<br>
`homepage`:   La página web oficial de la película<br>	
`id`:   ID de la pelicula	<br>
`imdb_id`:   IMDB ID de la pelicula	<br>
`original_language`:   Idioma original en la que se grabo la pelicula	<br>
`original_title`:   Titulo original de la pelicula	<br>
`overview`:   Pequeño resumen de la película	<br>
`popularity`:   Puntaje de popularidad de la película, asignado por TMDB (TheMoviesDataBase)	<br>
`poster_path`:   URL del póster de la película	<br>
`production_companies`:   Lista con las compañias productoras asociadas a la película	<br>
`production_countries`:   Lista con los países donde se produjo la película	<br>
`release_date`:   Fecha de estreno de la película	<br>
`revenue`:   Recaudación de la pelicula, en dolares	<br>
`runtime`:   Duración de la película, en minutos	<br>
`spoken_languages`:   Lista con los idiomas que se hablan en la pelicula	<br>
`status`:   Estado de la pelicula actual (si fue anunciada, si ya se estreno, etc)	<br>
`tagline`:   Frase celebre asociada a la pelicula	<br>
`title`:   Titulo de la pelicula	<br>
`video`:   Indica si hay o no un trailer en video disponible en TMDB	<br>
`vote_average`:   Puntaje promedio de reseñas de la pelicula	<br>
`vote_count`:   Numeros de votos recibidos por la pelicula, en TMDB	<br>


### `credits.csv`:

---                --------------  

  `cast`:    Elenco de la película en formato JSON<br>
  `crew`:   Equipo de producción de la película (Director, Productor, etc) en formato JSON<br>
    `id`:   ID de la película<br>

  
Finalmente, esta es la información de la base de datos lista (`dbSm.parquet`) para ser utilizada en el EDA y el entrenamiento del modelo para el Sistema de recomendación:


```python

<class 'pandas.core.frame.DataFrame'>
Int64Index: 44285 entries, 0 to 45458
Data columns (total 25 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   index                 44285 non-null  int64         
 1   id                    44285 non-null  int64         
 2   title                 44285 non-null  string        
 3   genres                44285 non-null  string        
 4   overview              44285 non-null  string        
 5   tagline               44285 non-null  string        
 6   spoken_languages      44285 non-null  string        
 7   original_language     44285 non-null  string        
 8   budget                44285 non-null  int64         
 9   revenue               44285 non-null  int64         
 10  popularity            44285 non-null  float64       
 11  release_date          44285 non-null  datetime64[ns]
 12  runtime               44285 non-null  int64         
 13  production_companies  44285 non-null  string        
 14  production_countries  44285 non-null  string        
 15  vote_average          44285 non-null  float64       
 16  vote_count            44285 non-null  int64         
 17  release_year          44285 non-null  int64         
 18  return                44285 non-null  float64       
 19  cast                  44285 non-null  string        
 20  director              44285 non-null  string        
 21  crew1                 44285 non-null  string        
 22  crew2                 44285 non-null  string        
 23  crew3                 44285 non-null  string        
 24  crew4                 44285 non-null  string        
dtypes: datetime64[ns](1), float64(3), int64(7), string(14)
memory usage: 8.8 MB

```



## Modelo de `Machine Learning` utilizado



Cosine Similarity


La **Similitud coseno** (Cosine Similarity) es una medida de similitud entre dos vectores no nulos definidos en un espacio de producto interno. Se mide por el coseno del ángulo entre los vectores; es decir, es el producto escalar de los vectores dividido por el producto de sus longitudes. Esto significa que la similitud del coseno no depende de las magnitudes de los vectores, sino solo de su ángulo.

|<img src = "https://www.tyrrell4innovation.ca/wp-content/uploads/2021/06/rsz_jenny_du_miword.png" height = 450> | <img src = "../_src/assets/RS.jpg" height = 300>|
|- |- |

La similitud coseno se utiliza comúnmente en análisis de datos para medir la similitud entre textos, o en nuestro caso, strings. Por ejemplo, en minería de texto cada palabra se asigna a una coordenada diferente y un texto se representa mediante el vector de los números de ocurrencias de cada palabra en el documento. La similitud coseno proporciona entonces una medida útil de cuán similares son dos strings en términos de su contenido, independientemente de la longitud de las mismas.
Nosotros convertiremos cada película en una especie de vector e intentaremos encontrar la similitud entre ellos utilizando las columnas `Title`, `Overview` y `Tagline` y dicho score de cuasi-similitud.

El modelo realmente anduvo muy bien, como podemos apreciar en la siguiente respuesta al input 'Batman':

```python

5629       Batman: Mystery of the Batwoman
4648            Batman: Under the Red Hood
6622    Batman Unlimited: Animal Instincts
6963                             Minutemen
2132                      Shanghai Knights
5192       Superman/Batman: Public Enemies
6027             Batman: Assault on Arkham
6237                     Batman vs Dracula
882                          Baby Geniuses
Name: title, dtype: string

```
Lamentablemente necesita casi 16 GB de Ram, imposible deployarlo en Render gratuitamente. La solución mas simple que se consideró fue disminuir el dataset a 2000 registros, pero según mi opinión eso va en contra de un buen recomendador. Por eso decidimos utilizar otro modelo de ML que no fuera tan Ram intensivo.



K-Nearest Neighbour


|<img src = "https://www.unite.ai/wp-content/uploads/2020/02/419px-KnnClassification.svg.png" height = 250> | |
|- |- |


A continuación probamos otro  de sistema de recomendación basado en la popularidad, donde se recomendaran películas segun su genero y popularidad entre multiples usuarios.

K-Nearest Neighbour (K-NN) es un algoritmo de aprendizaje automático que se puede utilizar tanto para tareas de regresión como de clasificación. Es un método no paramétrico, lo que significa que no asume ninguna distribución de los datos. Hace predicciones basándose en la proximidad o similitud de un nuevo punto de datos a un número elegido de puntos de datos del conjunto de entrenamiento.

En el caso de la clasificación, se asigna una etiqueta de clase en función de una votación mayoritaria, es decir, la etiqueta que está más frecuentemente representada alrededor de un punto de datos dado se utiliza. En el caso de la regresión, el promedio de los k vecinos más cercanos se toma para hacer una predicción sobre una clasificación.



