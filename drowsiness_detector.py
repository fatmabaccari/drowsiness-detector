import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Constantes
IMAGES_DIR = r'C:\Users\Fatma BACCARI\MLprojet\data\train\images'
LABELS_DIR = r'C:\Users\Fatma BACCARI\MLprojet\data\train\labels'
IMAGE_SIZE = (64, 64)


# Charger les données
def load_data(images_dir, labels_dir, image_size):
    X, y = [], []
    for img_file, lbl_file in zip(sorted(os.listdir(images_dir)), sorted(os.listdir(labels_dir))):
        img_path, lbl_path = os.path.join(images_dir, img_file), os.path.join(labels_dir, lbl_file)

        # Charger et prétraiter l'image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size).astype(np.float32) / 255.0
        X.append(img)

        # Charger l'étiquette
        with open(lbl_path, 'r') as f:
            y.append(int(f.readline().strip().split()[0]))

    return np.array(X), np.array(y)


X, y = load_data(IMAGES_DIR, LABELS_DIR, IMAGE_SIZE)

# Aplatir les images
X = X.reshape(X.shape[0], -1)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Affichage des premières images
num_images = 8
fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
for i in range(num_images):
    ax = axes[i]
    ax.imshow(X[i].reshape(IMAGE_SIZE), cmap='gray')  # Reshape pour obtenir une image 2D
    ax.set_title(f"Image {i+1}, Label: {'awake' if y[i] == 1 else 'drowsy'}")
    ax.axis('off')
plt.tight_layout()
plt.show()
print(f"Nombre total d'images : {len(X)}")
print(f"Nombre total des labels : {len(y)}")
num_awake = np.sum(y == 1)
num_drowsy = np.sum(y == 0)

print(f"Nombre d'images avec le label 'awake': {num_awake}")
print(f"Nombre d'images avec le label 'drowsy': {num_drowsy}")
print(f"Nombre d'images pour l'entraînement : {len(X_train)}")
print(f"Nombre d'images pour le test : {len(X_test)}")


# Modèle SVM
svm_params = {'kernel': ['linear', 'rbf','poly']}
svm_grid = GridSearchCV(SVC(), svm_params, cv=4, n_jobs=-1, verbose=2)
svm_grid.fit(X_train_scaled, y_train)
best_svm_model = svm_grid.best_estimator_

# Modèle MLP
mlp_params = {
    'hidden_layer_sizes': [(64, 32), (128, 64)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [500],
    'batch_size': [32,64]
}
mlp_grid = GridSearchCV(MLPClassifier(random_state=42), mlp_params, cv=4, n_jobs=-1, verbose=2)
mlp_grid.fit(X_train_scaled, y_train)
best_mlp_model = mlp_grid.best_estimator_

# Modèle KNN
knn_params = {'n_neighbors': [3, 5, 7,11,15], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=4, n_jobs=-1, verbose=2)
knn_grid.fit(X_train_scaled, y_train)
best_knn_model = knn_grid.best_estimator_


# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Précision": precision_score(y_test, y_pred, average='weighted'),
        "Rappel": recall_score(y_test, y_pred, average='weighted'),
        "Exactitude": accuracy_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm


# Evaluation des modèles
svm_metrics, svm_cm = evaluate_model(best_svm_model, X_test_scaled, y_test)
mlp_metrics, mlp_cm = evaluate_model(best_mlp_model, X_test_scaled, y_test)
knn_metrics, knn_cm = evaluate_model(best_knn_model, X_test_scaled, y_test)

# Affichage des matrices de confusion
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Drowsy', 'Awake'], yticklabels=['Drowsy', 'Awake'],
            ax=axs[0])
axs[0].set_title('SVM - Confusion Matrix')
axs[0].set_xlabel('Predicted')
axs[0].set_ylabel('Actual')

sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Drowsy', 'Awake'], yticklabels=['Drowsy', 'Awake'],
            ax=axs[1])
axs[1].set_title('MLP - Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('Actual')

sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Drowsy', 'Awake'], yticklabels=['Drowsy', 'Awake'],
            ax=axs[2])
axs[2].set_title('KNN - Confusion Matrix')
axs[2].set_xlabel('Predicted')
axs[2].set_ylabel('Actual')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Données à afficher
metrics = ['Précision', 'Rappel', 'Exactitude', 'F1-score']
svm_vals = [svm_metrics['Précision'], svm_metrics['Rappel'], svm_metrics['Exactitude'], svm_metrics['F1-score']]
mlp_vals = [mlp_metrics['Précision'], mlp_metrics['Rappel'], mlp_metrics['Exactitude'], mlp_metrics['F1-score']]
knn_vals = [knn_metrics['Précision'], knn_metrics['Rappel'], knn_metrics['Exactitude'], knn_metrics['F1-score']]

# Créer la figure et l'axe
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2  # Largeur des barres
index = np.arange(len(metrics))  # Position des barres sur l'axe X

# Barres pour chaque modèle avec des couleurs distinctes
bar_svm = ax.bar(index - bar_width, svm_vals, bar_width, label='SVM', color='#9467bd')  # Violet
bar_knn = ax.bar(index, knn_vals, bar_width, label='KNN', color='#ff7f0e')  # Orange
bar_mlp = ax.bar(index + bar_width, mlp_vals, bar_width, label='MLP', color='#1f77b4')  # Bleu clair

# Ajouter des valeurs sur les barres pour plus de clarté
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Décalage vertical des valeurs
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontweight='bold')

add_values(bar_svm)
add_values(bar_knn)
add_values(bar_mlp)

# Ajouter des labels et un titre
ax.set_xlabel('Métriques', fontsize=12)
ax.set_ylabel('Valeurs', fontsize=12)
ax.set_title('Comparaison des Métriques entre les Modèles', fontsize=14)
ax.set_xticks(index)
ax.set_xticklabels(metrics, fontsize=12)

# Légende positionnée dans le coin supérieur droit avec ajustement
ax.legend(title='Modèles', fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

# Améliorer l'apparence générale
ax.set_facecolor('whitesmoke')  # Fond clair

# Réajuster l'affichage
plt.tight_layout()

# Afficher le graphique
plt.show()


# Evaluation avec les scores et paramètres
def display_grid_search_results(grid_search, model_name):
    # Affichage des meilleurs paramètres
    print(f"\nMeilleurs paramètres pour {model_name} : {grid_search.best_params_}")

    # Affichage du meilleur score
    print(f"Meilleur score de {model_name}: {grid_search.best_score_}")

    # Affichage de la moyenne des scores pour toutes les grilles
    mean_scores = grid_search.cv_results_['mean_test_score']
    print(f"Moyenne des scores pour {model_name}: {mean_scores.mean():.4f}")




# Affichage des résultats des grilles de recherche
display_grid_search_results(svm_grid, "SVM")
display_grid_search_results(mlp_grid, "MLP")
display_grid_search_results(knn_grid, "KNN")

# Capture vidéo en temps réel
model_to_use = best_mlp_model

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement de la frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    flattened = resized.flatten() / 255.0
    processed_frame = scaler.transform(flattened.reshape(1, -1))

    # Prédiction avec le modèle choisi
    prediction = model_to_use.predict(processed_frame)
    label = 'Awake' if prediction[0] == 1 else 'Drowsy'
    color = (0, 255, 0) if prediction[0] == 1 else (0, 0, 255)

    # Afficher les résultats
    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Real-Time Drowsiness Detection', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
