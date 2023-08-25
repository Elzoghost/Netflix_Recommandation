import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données de notation de films à partir du dataset Netflix public
df = pd.read_csv('nf_prize_dataset.tar/nf_prize_dataset.tar.gz', header=None, names=['User', 'Movie', 'Rating', 'Date'], usecols=[0, 1, 2, 3])

# Créer une matrice utilisateur-produit
ratings_matrix = df.pivot_table(index=['User'], columns=['Movie'], values='Rating')

# Remplacer les valeurs manquantes par zéro
ratings_matrix = ratings_matrix.fillna(0)

# Calcul de la similarité cosinus entre les utilisateurs
similarite = cosine_similarity(ratings_matrix)

# Fonction de recommandation de films similaires à ceux aimés par un utilisateur donné
def recommander_films(utilisateur):
    # Trouver les films que l'utilisateur a aimés
    films_aimés = ratings_matrix.loc[utilisateur][ratings_matrix.loc[utilisateur] > 0].index
    # Trouver les utilisateurs similaires à l'utilisateur donné
    utilisateurs_similaires = np.argsort(similarite[ratings_matrix.index.get_loc(utilisateur),:])[::-1][1:]
    # Recommander des films aimés par les utilisateurs similaires mais pas encore vus par l'utilisateur donné
    recommandations = []
    for user in utilisateurs_similaires:
        films_aimés_similaires = ratings_matrix.iloc[user][ratings_matrix.iloc[user].index.isin(films_aimés)].index
        recommandations.extend(list(set(films_aimés_similaires) - set(films_aimés)))
        if len(recommandations) >= 5:
            break
    return recommandations

# Exemple de recommandation pour l'utilisateur 12345
print(recommander_films(12345))
