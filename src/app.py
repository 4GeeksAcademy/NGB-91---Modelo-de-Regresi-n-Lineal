from utils import db_connect
engine = db_connect()

# your code here
# Explore here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np

# VER BASE DE DATOS:

df_datos_sanitarios = pd.read_csv('../data/raw/demographic_health_data.csv')
df_datos_sanitarios
columnas = df_datos_sanitarios.columns.tolist()
pd.set_option('display.max_columns', None)
print(columnas)
print(df_datos_sanitarios.info())
print(df_datos_sanitarios.describe())
print(df_datos_sanitarios.isnull().sum())

#LIMPIEZA DE DATOS:
df_datos_sanitarios.copy()

# Eliminar columnas irrelevantes:

cols_to_drop = [
    'fips', 'COUNTY_NAME', 'STATE_NAME', 'STATE_FIPS', 'CNTY_FIPS',
    'POP_ESTIMATE_2018', 'Total Population', 'county_pop2018_18 and older',
    'CI90LBINC_2018', 'CI90UBINC_2018',
    'anycondition_Lower 95% CI', 'anycondition_Upper 95% CI',
    'Obesity_Lower 95% CI', 'Obesity_Upper 95% CI',
    'Heart disease_Lower 95% CI', 'Heart disease_Upper 95% CI',
    'COPD_Lower 95% CI', 'COPD_Upper 95% CI',
    'diabetes_Lower 95% CI', 'diabetes_Upper 95% CI',
    'CKD_Lower 95% CI', 'CKD_Upper 95% CI',
    '0-9', '19-Oct', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+',
    'White-alone pop', 'Black-alone pop', 'Native American/American Indian-alone pop',
    'Asian-alone pop', 'Hawaiian/Pacific Islander-alone pop', 'Two or more races pop',
    'Less than a high school diploma 2014-18',
    'High school diploma only 2014-18',
    "Some college or associate's degree 2014-18",
    "Bachelor's degree or higher 2014-18",
    'Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018',
    'Urban_rural_code',
    'N_POP_CHG_2018', 'GQ_ESTIMATES_2018', 'R_birth_2018', 'R_death_2018',
    'R_NATURAL_INC_2018', 'R_INTERNATIONAL_MIG_2018', 'R_DOMESTIC_MIG_2018',
    'R_NET_MIG_2018'
]
df_sanit_limpio = df_datos_sanitarios.drop(columns= cols_to_drop, errors='ignore')
df_sanit_limpio
df_sanit_limpio.info()
df_sanit_limpio.describe().T
df_sanit_limpio.isnull().sum().sort_values(ascending=False).head(20)

#DISTRIBUCIÓN DE LA VARIABLE:

y = df_sanit_limpio['diabetes_prevalence']
y
plt.figure(figsize=(8, 5))
sns.histplot(df_sanit_limpio['diabetes_prevalence'], kde=True, bins=30, color='indigo', edgecolor='black')
plt.title("Distribución de la prevalencia de la diabetes")
plt.xlabel("Prevalencia (%)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
#CORRELACIONES DE LA VARIABLE:
# CORRELACIONES:

correlations = df_sanit_limpio.corr(numeric_only=True)['diabetes_prevalence'].sort_values(ascending=False)
correlations
#ENTRENAMIENTO:
X = df_sanit_limpio.drop(columns=['diabetes_prevalence'])
y = df_sanit_limpio['diabetes_prevalence']

X = X.select_dtypes(include='number')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#REGRESIÓN LINEAL:
# LINEAL BÁSICA:

# Modelo:

lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicciones:

y_pred_lr = lr.predict(X_test)

# Evaluación:

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MSE (Linear Regression): {mse_lr:.2f}")
print(f"R2 Score (Linear Regression): {r2_lr:.4f}")
# LASSO: 

# Modelo:
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Predicción:
y_pred_lasso = lasso.predict(X_test)

# Evaluación:

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"MSE (Lasso Regression): {mse_lasso:.2f}")
print(f"R2 Score (Lasso Regression): {r2_lasso:.4f}")
print(f"Mejor alpha encontrado: {lasso.alpha_:.4f}")

# COMPARACIÓN ENTRE LOS DOS MODELOS:

plt.figure(figsize=( 8, 5))
plt.plot(y_test.values, label='Real', alpha=0.7, color='red')
plt.plot(y_pred_lr, label='Linear Reg.', alpha=0.7, color='darkblue')
plt.plot(y_pred_lasso, label='Lasso Reg.', alpha=0.7, color='darkgreen')
plt.legend()
plt.title('Comparación de predicciones')
plt.xlabel('Índice de muestra')
plt.ylabel('Prevalencia de diabetes')
plt.grid(True)
plt.show()
print("Linear y Lasso siguen un patrón general, no hay desvíos pero si que hay muchas líneas irregulares. Lasso si que parece menos fuerte que Linear que da algunos datos algo extremos. ")

#ANÁLISIS R2 EN FUNCIÓN DE LASSO: 
alphas = list(range(0, 21))  
r2_scores = []

for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    r2 = r2_score(y_test, lasso.predict(X_test))
    r2_scores.append(r2)

# Gráfico de evolución
plt.figure(figsize=(8, 5))
plt.plot(alphas, r2_scores, marker='o', color='indigo')
plt.title('Evolución del R² en función de alpha (Lasso)')
plt.xlabel('Alpha')
plt.ylabel('R²')
plt.grid(True)
plt.show()
print("A medida que el valor de alpha aumenta, el modelo Lasso se vuelve más estricto y reduce su capacidad para ajustarse bien a los datos, por eso el R² baja.")