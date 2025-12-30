# 🧠 Makine Öğrenmesi Yöntemlerinin Karşılaştırmalı Analizi: Sınıflandırma ve Regresyon

## 📌 Proje Hakkında
Bu proje, makine öğrenmesi literatüründe sıkça kullanılan **klasik** ve **modern** algoritmaların performanslarını karşılaştırmak amacıyla hazırlanmıştır.

Çalışma iki ana bölümden oluşmaktadır:

1️⃣ **Sınıflandırma (Classification):** Wifi sinyal verileri ile iç mekân konum tespiti  
2️⃣ **Regresyon (Regression):** Airfoil verileri ile gürültü seviyesi tahmini  

Amaç; farklı veri tiplerinde, farklı algoritma ailelerinin (vektör tabanlı, sinir ağı tabanlı, ağaç tabanlı) performanslarını ampirik olarak karşılaştırmaktır.

---

## 📂 Proje Yapısı

| Dosya | Açıklama |
|------|-----------|
| `classification.py` | C3 veri setinde **SVM** ve **ANN** modellerinin eğitimi, karşılaştırması ve görselleştirilmesi |
| `regression.py` | R2 veri setinde **XGBoost** ve **kNN** modellerinin eğitimi ve hata analizleri |
| `C3.mat` | Wifi sinyal sınıflandırma veri seti |
| `R2.mat` | Airfoil regresyon veri seti |
| `requirements.txt` | Gerekli Python kütüphaneleri |

---

## 🔬 Kullanılan Yöntemler

Her iki problemde de:

✔️ **3-Fold Cross Validation (Çapraz Doğrulama)**  
✔️ **StandardScaler (Ölçekleme)**  

uygulanmıştır.

### 🟢 Sınıflandırma — Wifi Verisi
**Karşılaştırılan modeller:**

- **SVM (RBF kernel)**
- **ANN (MLP — 64 ve 32 nöronlu iki gizli katman)**

---

### 🔵 Regresyon — Airfoil Verisi
**Karşılaştırılan modeller:**

- **XGBoost**
- **kNN**

Amaç: karmaşık boosting yapıları ile basit mesafe tabanlı yöntemlerin farkını görmek.

---

## ⚙️ Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

---

## 🚀 Çalıştırma Adımları

### 1️⃣ Sınıflandırma Analizi

```bash
python classification.py
```

**Görecekleriniz:**

- Katman bazlı başarı oranları
- Ortalama **Accuracy** ve **F1-score**
- **Confusion Matrix** görselleri

---

### 2️⃣ Regresyon Analizi

```bash
python regression.py
```

**Görecekleriniz:**

- **MAE** ve **SMAPE** değerleri
- Gerçek vs tahmin değerlerinin karşılaştırıldığı scatter grafikleri

---

## 📊 Sonuçların Yorumlanması

### 🔹 Sınıflandırma (C3)
- Accuracy → **1.0’a ne kadar yakınsa model o kadar başarılıdır**
- Confusion matrix → Koyu kareler köşegen üzerinde yoğunlaşmalıdır

### 🔹 Regresyon (R2)
- MAE ve SMAPE → **0’a yaklaştıkça hata azalır**
- Scatter grafikte → Noktaların **x = y** çizgisine yakın olması beklenir

---

## 🔁 Tekrarlanabilirlik

Projede:

```python
random_state = 42
```

kullanılmıştır. Böylece her çalıştırmada aynı sonuçlar elde edilir.

---

🎓 Bu proje **Makine Öğrenmesi dersi** kapsamında hazırlanmıştır.
