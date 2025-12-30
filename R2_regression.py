import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --------------------
# Veri yükleme
# --------------------
data = scipy.io.loadmat("R2.mat")
X = data["feat"]
y = data["lbl"].ravel()

print("R2 Veri Seti")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------
# k-Fold
# --------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# --------------------
# SMAPE fonksiyonu
# --------------------
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

models = [
    (SVR(), "SVR"),
    (LinearRegression(), "Linear Regression")
]

# --------------------
# Eğitim & Test
# --------------------
for model, name in models:
    mae_list = []
    smape_list = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, y_pred))
        smape_list.append(smape(y_test, y_pred))

    print(f"\n{name}")
    print("MAE:", np.mean(mae_list))
    print("SMAPE:", np.mean(smape_list))

# --------------------
# Gerçek vs Tahmin Grafikleri (HER MODEL)
# --------------------

models_for_plot = [
    (SVR(), "SVR"),
    (LinearRegression(), "Linear Regression")
]

for model, name in models_for_plot:
    model.fit(X, y)
    y_pred = model.predict(X)

    idx = np.random.choice(len(y), min(1000, len(y)), replace=False)

    plt.figure()
    plt.scatter(y[idx], y_pred[idx])
    plt.xlabel("Gerçek Değer")
    plt.ylabel("Tahmin")
    plt.title(f"Gerçek vs Tahmin - {name}")
    plt.show()

