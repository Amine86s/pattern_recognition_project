import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def run_regression():
    print("\n" + "="*50)
    print("   BÖLÜM 2: REGRESYON (R2 - Airfoil)")
    print("   Modeller: XGBoost vs kNN")
    print("="*50)

    # 1. Veri Yükleme
    filename = "R2.mat"
    try:
        data = scipy.io.loadmat(filename)
    except FileNotFoundError:
        print(f"HATA: '{filename}' bulunamadı!")
        return

    X = data["feat"]
    y = data["lbl"].ravel()
    print(f"Veri Yüklendi -> X: {X.shape}, y: {y.shape}")

    # 2. Ön İşleme (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Ayarlar
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def smape(y_true, y_pred):
        return 100 * np.mean(
            2 * np.abs(y_pred - y_true) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )

    # Modeller
    models = [
        (XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42), "XGBoost"),
        (KNeighborsRegressor(n_neighbors=5), "kNN")
    ]

    # 4. Eğitim & Test
    for model, name in models:
        mae_list = []
        smape_list = []

        print(f"\nModel Hesaplanıyor: {name}...")

        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae_list.append(mean_absolute_error(y_test, y_pred))
            smape_list.append(smape(y_test, y_pred))

        print(f"--> {name} Sonuçları:")
        print(f"MAE  : {np.mean(mae_list):.4f}")
        print(f"SMAPE: {np.mean(smape_list):.4f}%")

    # 5. Grafikler
    print("\n--- Grafikler Çiziliyor ---")
    for model, name in models:
        model.fit(X_scaled, y)
        y_pred_full = model.predict(X_scaled)

        # Çok veri varsa rastgele 1000 tanesini çiz (grafik kasmasın)
        if len(y) > 1000:
            idx = np.random.choice(len(y), 1000, replace=False)
            y_sample = y[idx]
            y_pred_sample = y_pred_full[idx]
        else:
            y_sample = y
            y_pred_sample = y_pred_full

        plt.figure(figsize=(6, 6))
        plt.scatter(y_sample, y_pred_sample, alpha=0.6, edgecolors='b')
        
        # İdeal x=y çizgisini ekle
        m = min(y_sample.min(), y_pred_sample.min())
        M = max(y_sample.max(), y_pred_sample.max())
        plt.plot([m, M], [m, M], 'r--', lw=3, label='İdeal (x=y)')
        
        plt.xlabel("Gerçek Değerler")
        plt.ylabel("Tahmin Edilen Değerler")
        plt.title(f"Regresyon: {name} (Gerçek vs Tahmin)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_regression()