import scipy.io

# C3 - Sınıflandırma
data_c3 = scipy.io.loadmat("C3.mat")
Xc = data_c3["feat"]
yc = data_c3["lbl"]

print("C3 veri seti:")
print("X shape:", Xc.shape)
print("y shape:", yc.shape)

# R2 - Regresyon
data_r2 = scipy.io.loadmat("R2.mat")
Xr = data_r2["feat"]
yr = data_r2["lbl"]

print("\nR2 veri seti:")
print("X shape:", Xr.shape)
print("y shape:", yr.shape)
