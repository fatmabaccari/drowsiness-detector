# Détecteur d'Éveil - Machine Learning

Ce projet détecte les signes de somnolence ("drowsiness") à partir de flux vidéo webcam, en utilisant des modèles de Machine Learning.

---

## 🚀 Fonctionnalités

- Entraînement et évaluation de plusieurs modèles (SVM, MLP, KNN) sur un dataset d’images de visages annotées  
- Prédiction en temps réel via webcam  
- Affichage dynamique d’un message "Awake" ou "Drowsy" sur la vidéo  

---

## 📁 Structure du projet

drowsiness-detector/
├── data/ # Images et labels
├── drowsiness_detector.py # Script principal
├── requirements.txt # Dépendances Python
├── README.md # Documentation (ce fichier)
└── .gitignore # Fichiers ignorés par git

yaml
Copier
Modifier

---

## ⚙️ Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/TON_UTILISATEUR/drowsiness-detector.git
cd drowsiness-detector
Créez et activez un environnement virtuel :

bash
Copier
Modifier
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux / MacOS
source venv/bin/activate
Installez les dépendances :

bash
Copier
Modifier
pip install -r requirements.txt
🚴‍♂️ Usage
Lancez le script principal :

bash
Copier
Modifier
python drowsiness_detector.py
Le programme affichera quelques images d'exemple

Puis entraînera les modèles et affichera des résultats

Enfin, ouvrira la webcam pour détecter l’état d’éveil en temps réel

Appuyez sur q pour quitter

📊 Détails techniques
Les images doivent être placées dans data/images/

Les labels (fichiers .txt) doivent être dans data/labels/

Les modèles utilisent scikit-learn avec recherche de paramètres (GridSearchCV)

Affichage des matrices de confusion et graphiques avec matplotlib et seaborn

✉️ Contact
Pour toute question : baccarifatma842003@gmail.com
