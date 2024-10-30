import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv') 
df.loc[:, 'MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0]) 
df.loc[:, 'LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.loc[:, 'SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']].copy()
y = df['SalePrice']
X.loc[:, :] = X.fillna(X.mean())
y = y.dropna()
X = X.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predicted_prices = pd.DataFrame({'Predicted Sale Price': predictions})
print(predicted_prices)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Actual vs Predicted Sale Price')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.grid()
plt.show()