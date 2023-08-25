# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Charger le dataset
df = pd.read_csv('netflix_titles.csv')

# Comprendre quel contenu est disponible dans différents pays
countries = df['country'].str.split(',', expand=True).stack().value_counts()
print(countries)

# Identifier un contenu similaire en faisant correspondre des fonctionnalités textuelles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
df['description'] = df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

print(get_recommendations('Breaking Bad'))

from collections import Counter

def get_network(df, column):
    pairs = {}
    for vals in df[column]:
        vals = vals.split(',')
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                pair = ','.join(sorted((vals[i], vals[j]), key=str))
                if pair in pairs:
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1
    count = Counter(pairs).most_common(10)
    return pd.DataFrame(count, columns=['pair', 'count'])

network_df = get_network(df, 'cast')
print(network_df)


# Est-ce que Netflix se concentre davantage sur les émissions de télévision que sur les films ces dernières années ?
df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year
sns.countplot(x='year_added', hue='type', data=df)
plt.show()
