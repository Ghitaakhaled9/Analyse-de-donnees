import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

lire = pd.read_csv(r"C:\Users\Admin\OneDrive\Bureau\ACP\Automobile_data.csv")
final=lire[['highway-mpg','engine-size','horsepower','curb-weight']]
print(final)
for x,row in final.iterrows():
    if row['horsepower']=='?' :
        final.drop(x,inplace=True)
print(final)
final=final.astype(float)
matrice = (final-final.mean())/final.std()
matrice=matrice.to_numpy()
print(matrice)

covariance_matrix=np.cov(matrice.T)
valp,vecp=np.linalg.eig(covariance_matrix)

print(vecp)
print(valp)

p1 = matrice.dot(vecp[0])
p2 = matrice.dot(vecp[1])
p3 = matrice.dot(vecp[2])
p4 = matrice.dot(vecp[3])
df = pd.DataFrame({'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4})
sns.scatterplot(data=df, x='p2', y='p1')

for x,row in lire.iterrows():
    if row['price']=='?' :
        lire.drop(x,inplace=True)

price_col = lire.pop("price")

df = df.assign(new_feature=price_col)
sns.scatterplot(data=df, x='p1', y='p2', hue='new_feature')
corr_mat = np.corrcoef(matrice.T)
valc, vecc = np.linalg.eig(corr_mat)

fig, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.5)
sns.set_style('white')
circle = plt.Circle((0, 0), radius=1, color='black', fill=False)
ax.add_artist(circle)
for i in range(len(vecc)):
    ax.arrow(0, 0, vecc[i, 0], vecc[i, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    ax.text(vecc[i, 0]*1.2, vecc[i, 1]*1.2, final.columns[i], color='k', ha='center', va='center')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_aspect('equal', adjustable='box')
plt.show()

n = matrice.shape[0]
indices = np.random.randint(0, n, size=10)
quality = []
for idx in indices:
    projected_point = matrice[idx].dot(vecp[:, :2])
    reconstructed_point = projected_point.dot(vecp[:, :2].T)
    error = np.linalg.norm(matrice[idx] - reconstructed_point)
    quality.append(1 - error**2/np.linalg.norm(matrice[idx])**2)

print(f"Projection quality for 10 random points: {quality}")

#le cercle de corrélation montre que les variables "highway-mpg" et "curb-weight" 
#sont fortement corrélées avec le premier composant principal, tandis que "engine-size" 
#et "horsepower" sont plus fortement corrélées avec le deuxième composant principal.
#Les valeurs de qualité de projection suggèrent que l'espace PCA bidimensionnel
#préserve une quantité relativement élevée d'informations pour les 10 points de données sélectionnés au hasard.
