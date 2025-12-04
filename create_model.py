import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Créer un modèle factice
X_dummy = np.array([[50, 2, 1], [100, 3, 2], [150, 4, 2]])
y_dummy = np.array([200000, 350000, 500000])

model = LinearRegression()
model.fit(X_dummy, y_dummy)

# Sauvegarder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle factice créé : model.pkl")