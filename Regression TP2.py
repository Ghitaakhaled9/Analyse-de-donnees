from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

lire1 = pd.read_csv(r"C:\Users\hp\OneDrive\Bureau\ACP\Automobile_data.csv")

df1 = lire1[['highway-mpg','engine-size','horsepower','curb-weight']]
df2 = lire1['price']


df1 = df1.replace('?', 0)
df2=df2.replace('?', 0)

df1_train, df1_test, df2_train, df2_test = train_test_split(df1, df2, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(df1_train, df2_train)

df2_pred = model.predict(df1_test)

print("Coefficients de régression : ", model.coef_)
print("Erreur quadratique moyenne : ", mean_squared_error(df2_test, df2_pred))
print("Score R² : ", r2_score(df2_test, df2_pred))

df= pd.DataFrame({'Valeur réelle': df2_test, 'Prédiction': df2_pred})
print(df)
