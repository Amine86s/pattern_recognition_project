import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --------------------
# Veri yükleme
# --------------------
data = scipy.io.loadmat("C3.mat")
X = data["feat"]
y = data["lbl"].ravel()

print("C3 Veri Seti")
print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------
# k-Fold
# --------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)

models = [
    (SVC(), "SVM"),
    (KNeighborsClassifier(n_neighbors=5), "kNN")
]

# --------------------
# Eğitim & Test
# --------------------
for model, name in models:
    acc_list = []
    f1_list = []
    cm_total = np.zeros((len(np.unique(y)), len(np.unique(y))))

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, average="macro"))
        cm_total += confusion_matrix(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", np.mean(acc_list))
    print("F1-Score:", np.mean(f1_list))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.show()
