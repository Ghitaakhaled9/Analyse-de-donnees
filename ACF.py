import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

couleur_des_cheveux = ["chatains","Roux","Blond"]
couleur_des_yeux = ["Marrons","Noisette","Verts","Bleus"]

matrice = pd.DataFrame({
    'Chatains': [119, 54, 29, 84],
    'Roux': [26, 14, 14, 17],
    'Blonds': [7, 10, 16, 94],
}, index=['Marrons', 'Noisette', 'Verts', 'Bleus'])

print (matrice)

#calcul du nombre totale des observations :
nbrTotale = matrice.sum().sum()



# Calculer le tableau des fréquences relatives
fij = matrice / nbrTotale
fij_mat = np.array(fij)
fij_mat = np.array(fij)


#calcul de la matrice des Fréquences relatives et marges
#somme en ligne
row_margins = np.sum(fij_mat, axis=1).reshape(-1,1)
#ajout a notre matrice
fij_mat = np.concatenate((fij_mat, row_margins), axis=1)
#somme en colonnes
col_margins = np.sum(fij_mat, axis=0).reshape(1,-1)
# Concaténation de la matrice de marges des colonnes avec la matrice fij_mat
fij_mat = np.concatenate((fij_mat, col_margins), axis=0)
#affichage
print(fij_mat)


# Calculer les marges pour les lignes et les colonnes
row_totals = matrice.sum(axis=1)
col_totals = matrice.sum(axis=0)

# Calculer les profils de ligne
L = matrice.values / row_totals.values.reshape(-1, 1)

# Afficher les profils de ligne
print("Profils de ligne:")
print(L)



# Calculer les profils de colonne
C = matrice.values / col_totals.values.reshape(1, -1)


# Afficher les profils de colonne
print("Profils de colonne:")
print(C)

total_matrice = np.sum(matrice)
print(total_matrice)


correspondenceMatrice = np.divide(matrice,total_matrice)
print(correspondenceMatrice)

rowTotals = np.sum(correspondenceMatrice,axis=1)
print(rowTotals)
columnTotals = np.sum(correspondenceMatrice,axis=0)
print(columnTotals)


independenceMatrice = np.outer(rowTotals, columnTotals)
print (independenceMatrice)

chiSquaredStatistic = total_matrice*np.sum(np.square(correspondenceMatrice-independenceMatrice)/independenceMatrice)
print(chiSquaredStatistic)

statistic,p_value,degreeOfFreedom,expectedValues = chi2_contingency(matrice)
print(statistic)

print(p_value)

print(degreeOfFreedom)

print(expectedValues)

standardizedResiduals = np.divide((correspondenceMatrice-independenceMatrice),np.sqrt(independenceMatrice))
u,s,vh = np.linalg.svd(standardizedResiduals, full_matrices=False)
print(u)

print(s)

print(vh)

stdRows = np.zeros((u.shape[0],u.shape[1]))
for i in range(u.shape[0]):
    stdRows[i] = np.divide(u[i],np.sqrt(rowTotals[i]))
print(stdRows)    
rowCoordinates = np.dot(stdRows,np.diag(s)) 

columntotals = np.matrix ([[0.59090909],[0.14669421],[0.26239669]])
stdCols = np.zeros((vh.shape[0],vh.shape[1])) 
for i in range(u.shape[1]):
    stdCols[i] = np.divide(vh[i],np.sqrt(columntotals[i]))
print(stdCols)    
colCoordinates = np.dot(stdCols,np.diag(s))

dfFirstTwoComponentsR = pd.DataFrame(data=[l[0:2] for l in rowCoordinates], columns=['dim1', 'dim2'], index=couleur_des_yeux)
dfFirstTwoComponentsC = pd.DataFrame(data=[l[0:2] for l in colCoordinates], columns=['dim1', 'dim2'], index=couleur_des_cheveux)
dfFirstTwoComponents = pd.concat([dfFirstTwoComponentsR, dfFirstTwoComponentsC])

# see what our final data looks like 
#see what our final data looks like 
print(dfFirstTwoComponents) 
points = couleur_des_yeux + couleur_des_cheveux

ax = sns.scatterplot(data=dfFirstTwoComponentsR, x='dim1', y='dim2', hue=couleur_des_yeux, color='.2')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.get_legend().set_visible(False)
for label in couleur_des_yeux + couleur_des_cheveux:
    plt.annotate(label, (dfFirstTwoComponents.loc[label, 'dim1'], dfFirstTwoComponents.loc[label, 'dim2']),
                 horizontalalignment='center', verticalalignment='center', size=11)
plt.scatter([0]*len(couleur_des_cheveux), range(len(couleur_des_cheveux)), color='.2')
plt.yticks(range(len(couleur_des_cheveux)), couleur_des_cheveux)
plt.show()
