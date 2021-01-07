import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Data set
df = pd.read_csv("housing.csv")
df.head()

# See the information of dataset
df.info()

df.dropna(inplace=True)  # Keep the DataFrame with valid entries in the same variable.
df = df.drop('ocean_proximity', axis=1)  # drop the label='ocean_proximity' in columns
print(df.columns)  # See the list of columns

# Show the hist plot of median_house_value with matlab function
p1 = plt.figure(figsize=(8,4))
sns.distplot(df['median_house_value'])

p2 = plt.figure(figsize=(12,8))
sns.scatterplot(x='longitude',y='latitude',data=df,hue='median_house_value')



p3 = plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)



# Label separation
x = df.drop('median_house_value',axis=1)
y = df['median_house_value']

# Split the train, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Rescale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.layers import Dropout
from keras import metrics

model = Sequential()
input_shape = X_train[0].shape

model.add(Dense(8, activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))

model.add(Dense(6, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(3, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error', metrics=[metrics.mae]) # Other performance metrics are not meaningful. So we should only use MSE or MAE

from keras.callbacks import EarlyStopping
#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

hist = model.fit(x=X_train, y=y_train.values, validation_data=(X_val,y_val.values), batch_size=128, epochs=200)

print(hist.history.keys())
#pyplot.plot(hist.history['loss'])
#pyplot.plot(hist.history['val_loss'])
#pyplot.legend
#pyplot.show()

losses = pd.DataFrame(hist.history)
losses.plot()
plt.draw()

from sklearn.metrics import mean_squared_error, mean_absolute_error
prediction = model.predict(X_test)
print("Mean abs Error: ", mean_absolute_error(y_test, prediction))
print("Mean squared Error: ", mean_squared_error(y_test, prediction))
print("Root of Mean squared Error: ", np.sqrt(mean_squared_error(y_test, prediction)))

plt.show()