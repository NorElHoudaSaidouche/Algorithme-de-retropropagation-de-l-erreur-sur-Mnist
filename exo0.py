"""
TP 1 - Algorithme de rétro-propagation de l’erreur
Exercice 0 : Visualisation de quelques images de la base
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""

# Importation des bibiothèques
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Chargement des données MNIST, et création de l'ensemble d'apprentissage et l'ensemble de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Affichage
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10, 20, i + 1)
    plt.imshow(X_train[i, :].reshape([28, 28]), cmap='gray')
    plt.axis('off')
plt.show()
