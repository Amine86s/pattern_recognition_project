import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1️⃣ VERİYİ YÜKLE
# -----------------------------
data = scipy.io.loadmat("R2.mat")

X = data["feat"]
y = data["lbl"].ravel()   # (1503, 1) → (1503,)

print("R2 veri seti:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 2️⃣ SMAPE FONKSİYONU
# -----------------------------
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

# -----------------------------
# 3️⃣ 3-FOLD CROSS VALIDATION
# -----------------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)

knn_mae, knn_smape = [], []
svr_mae, svr_smape = [], []

# Son fold için grafik çizmek adına saklayacağız
X_test_last, y_test_last = None, None
y_pred_knn_last, y_pred_svr_last = None, None

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # -------------------------
    # 4️⃣ kNN REGRESSOR
    # -------------------------
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    knn_mae.append(mean_absolute_error(y_test, y_pred_knn))
    knn_smape.append(smape(y_test, y_pred_knn))

    # -------------------------
    # 5️⃣ SVR
    # -------------------------
    svr = SVR(kernel="rbf")
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)

    svr_mae.append(mean_absolute_error(y_test, y_pred_svr))
    svr_smape.append(smape(y_test, y_pred_svr))

    # Grafik için son fold'u sakla
    X_test_last = X_test
    y_test_last = y_test
    y_pred_knn_last = y_pred_knn
    y_pred_svr_last = y_pred_svr

# -----------------------------
# 6️⃣ ORTALAMA SONUÇLAR
# -----------------------------
print("\n--- REGRESYON SONUÇLARI (3-Fold CV) ---")

print("kNN Regressor:")
print("MAE :", np.mean(knn_mae))
print("SMAPE :", np.mean(knn_smape))

print("\nSVR:")
print("MAE :", np.mean(svr_mae))
print("SMAPE :", np.mean(svr_smape))

# -----------------------------
# 7️⃣ GERÇEK vs TAHMİN GRAFİKLERİ
# -----------------------------
plt.figure()
plt.scatter(y_test_last, y_pred_knn_last)
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin")
plt.title("Gerçek vs Tahmin - kNN Regressor")
plt.show()

plt.figure()
plt.scatter(y_test_last, y_pred_svr_last)
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin")
plt.title("Gerçek vs Tahmin - SVR")
plt.show()
