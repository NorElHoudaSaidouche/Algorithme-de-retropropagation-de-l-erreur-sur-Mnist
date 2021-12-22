"""
TP 1 - Algorithme de rétro-propagation de l’erreur
Exercice 1 : Régression Logistique
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""

# Importation des bibliothèques
from keras.utils import np_utils
from keras.datasets import mnist
from divers import forward, backward, accuracy
import numpy as np

# Chargement des données MNIST, et création de l'ensemble d'apprentissage et l'ensemble de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalisation du dataset
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convertir les vecteurs de classe en matrices de classe binaires
K = 10  # Nombre de classe
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

# Nombre d'images pour l'ensemble d'apprentissage
N = X_train.shape[0]
# Taille de l'image (28*28 = 784 pixels)
d = X_train.shape[1]
# Poids
W = np.zeros((d, K))
# Biais
b = np.zeros((1, K))
# Nombre d'itérations
numEp = 20
# Taux d'apprentissage
eta = 1e-1
# Taille du batch
batch_size = 100
# Nombre de batches
nb_batches = int(float(N) / batch_size)

for epoch in range(numEp):
    for ex in range(nb_batches):
        print(f'======= Epoch {epoch}/{numEp} ======= Batch {ex}/{nb_batches}')
        # Récupération des données du batch en cours

        X_batch = X_train[ex * batch_size:(ex + 1) * batch_size]
        Y_batch = Y_train[ex * batch_size:(ex + 1) * batch_size]

        # FORWARD PASS : calculer la prédiction avec les paramètres actuels
        Y_pred = forward(X_batch, W, b)
        # BACKWARD PASS : calculer les gradients pour W et b
        # Mise à jour des paramètres W et b par descente de gradient
        W, b = backward(X_batch, Y_pred, Y_batch, eta, W, b)

acc = accuracy(W, b, X_test, Y_test)
print(f'Accuracy : {acc}')
