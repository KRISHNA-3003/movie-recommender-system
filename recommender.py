import pandas as pd

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep useful columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

print(movies.head())

# Remove missing values
movies.dropna(inplace=True)

print("Total movies:", movies.shape)

import ast

# Convert string to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
#Handle cast
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)
#Handle director from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
#Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())
#Combine all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
#Create final dataset
new_df = movies[['movie_id','title','tags']]
#Convert list → string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
#Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
#Print result
print(new_df.head())

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
#BUild Recommendation
def recommend(movie):
    movie = movie.lower()
    
    if movie not in new_df['title'].str.lower().values:
        print("Movie not found! Try correct name.")
        return
    
    movie_index = new_df[new_df['title'].str.lower() == movie].index[0]
    
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("\nRecommended movies:\n")
    for i in movies_list:
        print("-", new_df.iloc[i[0]].title)
movie = input("Enter movie name: ")
recommend(movie)