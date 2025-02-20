import pandas as pd
import pickle

with open("modelo.pkl", "rb") as f:
  best_rf = pickle.load(f)

columns = ['main_nominations', 'nominations', 'winner_gg', 'imdb_rating',
       'imdb_popularity', 'revenue', 'runtime', 'budget', 'release_month',
       'profit', 'War', 'Adventure', 'Music', 'Fantasy', 'Science Fiction',
       'Action', 'Romance', 'Drama', 'History', 'Family', 'Animation',
       'Comedy', 'Western', 'Thriller', 'Crime', 'Horror', 'Mystery']

num = columns[:10]
gen = columns[10:]

def novo_ano():
  df = {}  
  for i in range(10):
    nome = input(f'Filme {i+1}: ')
    film = []

    for j in num:
      film.append(input(f'{j} de {nome}: '))
    
    k = 0
    cont = 1
    genres = []
    print(f'Gêneros: {gen}')
    while k != '':
      genres.append(k)
      k = input(f'Gênero {cont} de {nome}: ')
      cont += 1
    genres.pop(0)

    for l in gen:
      if l in genres:
        film.append(1)
      else:
        film.append(0)

    df[nome] = film
      
  return(df)

criacao = novo_ano()

df = pd.DataFrame(criacao).transpose()

df.columns = columns

df['revenue'] = df['revenue'].astype(int)
df['budget'] = df['budget'].astype(int)

df['profit'] = df['revenue'] - 2*df['budget']

df = df.astype({col: 'int' for col in df.columns if col != 'imdb_rating'})

df['winner_gg'] = df['winner_gg'].astype(bool)
df['imdb_rating'] = df['imdb_rating'].astype(float)

y_pred = best_rf.predict_proba(df)[:,1]

y = sum(y_pred)

for i in range(len(y_pred)):
    y_pred[i] = y_pred[i]/y

resultado = pd.DataFrame({
    'Filme': df.index,
    'Probabilidade': y_pred
})

resultado = resultado.sort_values(by='Probabilidade', ascending=False).reset_index(drop = True)

resultado['Probabilidade'] = [f'{round(100*y, 2)}%' for y in resultado['Probabilidade']]

resultado.index = resultado.index + 1

print(resultado)