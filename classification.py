import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def run_classification():
    print("\n" + "="*50)
    print("   BÖLÜM 1: SINIFLANDIRMA (C3 - Wifi)")
    print("   Modeller: SVM vs ANN (Yapay Sinir Ağı)")
    print("="*50)

    # 1. Veri Yükleme
    filename = "C3.mat"
    try:
        data = scipy.io.loadmat(filename)
    except FileNotFoundError:
        print(f"HATA: '{filename}' bulunamadı! Dosyanın kodla aynı klasörde olduğundan emin olun.")
        return

    X = data["feat"]
    y = data["lbl"].ravel()
    print(f"Veri Yüklendi: {X.shape}")

    # 2. Ön İşleme (ANN ve SVM için Scaling ŞARTTIR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modeller
    models = [
        (SVC(kernel='rbf'), "SVM"),
        (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), "ANN (MLP)")
    ]

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # 4. Eğitim & Test
    for model, name in models:
        acc_list = []
        f1_list = []
        cm_total = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

        print(f"\nModel eğitiliyor: {name} (Lütfen bekleyin)...")

        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred, average="weighted"))
            cm_total += confusion_matrix(y_test, y_pred)

        print(f"--> {name} Sonuçları:")
        print(f"Accuracy: {np.mean(acc_list):.4f}")
        print(f"F1-Score: {np.mean(f1_list):.4f}")

        # Hata Matrisi Çizimi
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Hata Matrisi - {name}")
        plt.show()

if __name__ == "__main__":
    run_classification()