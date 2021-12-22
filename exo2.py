"""
TP 1 - Algorithme de rétro-propagation de l’erreur
Exercice 2 : Perceptron multi-couches (MLP)
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""

# Importation des bibliothèques
from keras.utils import np_utils
from keras.datasets import mnist
from divers import *

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

# Convertion des vecteurs de classe en matrices de classe binaires
K = 10  # Nombre de classe
L = 50
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)
# Taille de l'image (28*28 = 784 pixels)
d = X_train.shape[1]
# Nombre d'images pour l'ensemble d'apprentissage
N = X_train.shape[0]

# Initialisation des poids
# Distribution normal
#Wy = np.random.normal(0, 0.1, (L, K))
#by = np.random.normal(0, 0.1, (1, K))
#Wh = np.random.normal(0, 0.1, (d, L))
#bh = np.random.normal(0, 0.1, (1, L))

# Xavier
sigma_Xav = 1/np.sqrt(784)
Wy = np.random.normal(0, sigma_Xav,(L,K))
by = np.random.normal(0, sigma_Xav,(1,K))
Wh = np.random.normal(0, sigma_Xav,(d,L))
bh = np.random.normal(0, sigma_Xav,(1,L))

# Nombre d'itérations
numEp = 100
# Taux d'apprentissage
eta = 1.0
# Taille du batch
batch_size = 100
# Nombre de batches
nb_batches = int(float(N) / batch_size)

for epoch in range(numEp):
    for ex in range(nb_batches):
        print(f'======= Epoch {epoch}/{numEp} ======= Batch {ex}/{nb_batches}')
        # Récupération des données du batch en cours

        X_batch = X_train[ex * batch_size:(ex + 1) * batch_size, :]
        Y_batch = Y_train[ex * batch_size:(ex + 1) * batch_size, :]

        # FORWARD PASS : calculer la prédiction avec les paramètres actuels
        Y_pred, H = forward_mlp(X_batch, Wh, bh, Wy, by)
        Wy_n, by_n, Wh_n, bh_n = backward_mlp(X_batch, H, Y_pred, Y_batch, eta, Wh, bh, Wy, by)
        Wy, by, Wh, bh = Wy_n, by_n, Wh_n, bh_n

acc = accuracy_mlp(Wh, bh, Wy, by, X_test, Y_test)
print(f'Accuracy : {acc}')
