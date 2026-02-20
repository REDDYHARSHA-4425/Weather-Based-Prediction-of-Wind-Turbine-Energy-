import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample dataset
data = {
    "wind_speed": [5,6,7,8,9,10,11,12],
    "temperature": [25,24,23,22,21,20,19,18],
    "air_density": [1.2,1.19,1.18,1.17,1.16,1.15,1.14,1.13],
    "power_output": [200,250,300,350,400,450,500,550]
}

df = pd.DataFrame(data)

X = df[["wind_speed","temperature","air_density"]]
y = df["power_output"]

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model created successfully!")
