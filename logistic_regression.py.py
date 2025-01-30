import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri seti
data = {
    'Not': [65, 80, 75, 50, 90],
    'Ders_Calisma_Saati': [5, 7, 6, 3, 8],
    'Gecti': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Bağımsız ve bağımlı değişkenleri ayırma
X = df[['Not', 'Ders_Calisma_Saati']]
Y = df['Gecti']

# Eğitim ve test verilerini ayırma (%80 eğitim, %20 test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (Lojistik Regresyon daha iyi çalışsın diye)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli oluştur ve eğit
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Tahmin yap
Y_pred = model.predict(X_test_scaled)

# Doğruluk oranını hesapla
accuracy = accuracy_score(Y_test, Y_pred)
print('Doğruluk oranı:', accuracy)
