import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. GENERAREA DATELOR

def generate_factory_data(samples=1000, noise_level=0.1):
    """
    Simulăm date de senzori: 
    X = [Temperatura, Vibrații]
    y = [Status Motor] (0 = OK, 1 = Defect Iminent)
    """
    # Temperatura normala ~60 grade, Vibratii ~0.5
    X = np.random.rand(samples, 2) 
    
    # Regula a naturii (pe care AI-ul trebuie sa o invețe):
    # Daca (Temp > 0.8) SAU (Vibratii > 0.8) => Defect (1)
    y = ((X[:, 0] > 0.8) | (X[:, 1] > 0.8)).astype(int)
    
    # Adaugam zgomot de senzor
    X += np.random.normal(0, noise_level, X.shape)
    return X, y

# Cream datele pentru 2 "fabrici" izolate
print("Generare date fabrici...")
X_factory_A, y_factory_A = generate_factory_data(samples=1000)
X_factory_B, y_factory_B = generate_factory_data(samples=1000)


# 2. DEFINIREA MODELULUI AI (Arhitectura)

def create_model():
    model = models.Sequential([
        layers.Input(shape=(2,)),       # 2 Senzori
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Probabilitate defect
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. PRIVACY MECHANISM (Differential Privacy Lite)

def add_privacy_noise(weights, noise_scale=0.01):
    """
    Adaugă zgomot aleatoriu peste parametrii modelului.
    Asta previne "reverse engineering" al datelor originale.
    """
    noisy_weights = []
    for w in weights:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=w.shape)
        noisy_weights.append(w + noise)
    return noisy_weights


# 4. SIMULAREA FEDERATED LEARNING (Bucla Principala)

# Initializam modelul global
global_model = create_model()
global_weights = global_model.get_weights()

print("\nÎncepe antrenarea Federată (5 Runde)...")

# Modele locale (temporare)
model_A = create_model()
model_B = create_model()

accuracies = []

for round_num in range(1, 6):
    print(f"--- Runda {round_num} ---")
    
    # a) Serverul trimite modelul la "fabrici" 
    model_A.set_weights(global_weights)
    model_B.set_weights(global_weights)
    
    # b) Antrenare LOCALA (Datele nu pleaca nicaieri)
    # "Fabrica" A antreneaza pe datele ei
    model_A.fit(X_factory_A, y_factory_A, epochs=5, verbose=0, batch_size=32)
    # "Fabrica" B antreneaza pe datele ei
    model_B.fit(X_factory_B, y_factory_B, epochs=5, verbose=0, batch_size=32)
    
    # c) Extragem noile cunostinte (weights)
    weights_A = model_A.get_weights()
    weights_B = model_B.get_weights()
    
    # d) Aplicăm PRIVACY (Criptare prin zgomot)
    # Inainte sa trimita la server, fabricile "murdaresc" putin datele matematice
    secure_weights_A = add_privacy_noise(weights_A, noise_scale=0.005)
    secure_weights_B = add_privacy_noise(weights_B, noise_scale=0.005)
    
    # e) AGREGARE (Serverul face media)
    # Federated Averaging: Media aritmetica a parametrilor
    new_global_weights = []
    for w_a, w_b in zip(secure_weights_A, secure_weights_B):
        new_global_weights.append((w_a + w_b) / 2)
        
    # Actualizam modelul global
    global_weights = new_global_weights
    global_model.set_weights(global_weights)
    
    # Evaluam modelul global (pe un set de test neutru)
    X_test, y_test = generate_factory_data(100)
    loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(acc)
    print(f"Acuratețea Globală după runda {round_num}: {acc*100:.2f}%")

# 5. VIZUALIZARE REZULTATE

plt.plot(range(1, 6), accuracies, marker='o', linestyle='-', color='b')
plt.title('Performanța Modelului Federat (Privacy-Preserved)')
plt.xlabel('Runde de Comunicare')
plt.ylabel('Acuratețe')
plt.grid(True)
plt.show()

print("\nCONCLUZIE:")
print("Modelul a învățat 'regula defectelor' din două locații diferite,")
print("fără să vadă niciodată datele brute ale fabricilor și folosind Differential Privacy.")