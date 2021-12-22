"""
TP 1 - Algorithme de rétro-propagation de l’erreur
Implémentation des fonctions
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""

# Importation des bibliothèques
import numpy as np


# Implémantation de la fonction d'activation softmax
def softmax(x):
    e = np.exp(x)
    return (e.T / np.sum(e, axis=1)).T


# Implémentation de la fonction : FORWARD PASS
def forward(x_batch, w, b):
    # calculer la prédiction pour un batch de données
    pred = (x_batch @ w) + b
    y_pred = softmax(pred)
    return y_pred


# Implémentation de la fonction : BACKWARD PASS
def backward(x_batch, y_pred, y_true, learning_rate, w, b):

    n = len(x_batch)
    delta = y_pred - y_true
    # Calcul des gradients
    grad_w = 1 / n * (x_batch.T @ delta)
    grad_b = 1 / n * np.sum(delta, axis=0)
    # Mise à jour les paramètres par descente de gradient
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new


# Implémentation de la fonction accurancy
def accuracy(w, b, images, labels):
    pred = forward(images, w, b)
    return np.where(pred.argmax(axis=1) != labels.argmax(axis=1), 0., 1.).mean() * 100.0


# Implémentation de la fonction d'activation sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Implémentation de la fonction forward pour le MLP
def forward_mlp(x_batch, wh, bh, wy, by):
    U = x_batch @ wh + bh
    H = sigmoid(U)
    # Prédiction
    pred = forward(H, wy, by)
    return pred, H


# Implémentation de la fonction backward pour le MLP
def backward_mlp(x_batch, h, y_pred, y_true, learning_rate, wh, bh, wy, by):
    n = len(x_batch)
    delta = y_pred - y_true
    grad_wy = 1 / n * (h.T @ delta)
    grad_by = 1 / n * np.sum(delta, axis=0)

    delta_h = delta.dot(wy.T) * (h * (1 - h))
    grad_wh = 1 / n * (x_batch.T.dot(delta_h))
    grad_bh = 1 / n * np.sum(delta_h, axis=0)
    # Mise à jour les paramètres par descente de gradient
    wy_n = wy - learning_rate * grad_wy
    by_n = by - learning_rate * grad_by
    wh_n = wh - learning_rate * grad_wh
    bh_n = bh - learning_rate * grad_bh

    return wy_n, by_n, wh_n, bh_n


# Implémentation de la fonction accuracy pour le MLP
def accuracy_mlp(wh, bh, wy, by, images, labels):
    pred, h = forward_mlp(images, wh, bh, wy, by)
    return np.where(pred.argmax(axis=1) != labels.argmax(axis=1), 0., 1.).mean()*100.0
