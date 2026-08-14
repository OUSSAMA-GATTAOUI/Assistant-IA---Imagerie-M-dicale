# Assistant IA — Imagerie Médicale

Assistant intelligent d'aide au diagnostic par imagerie médicale (radiographie thoracique et IRM cérébrale), basé sur DenseNet121 et Grad-CAM, avec une interface web Streamlit.

## Contenu du dépôt

- `app.py` — application web Streamlit (dépôt d'image, classification, Grad-CAM, génération de compte-rendu PDF).
- `training_notebook.ipynb` — notebook d'entraînement des modèles de classification (type d'image, poumons, cerveau).
- `requirements.txt` — dépendances Python nécessaires à l'exécution de l'application.

## Notebooks & données non versionnés

Pour des raisons de taille et de confidentialité, certains éléments ne sont **pas inclus** dans ce dépôt :

- **Notebook de validation** — contient l'évaluation détaillée des modèles (matrices de confusion, courbes, tests sur données externes). [À COMPLÉTER — lien Drive/Kaggle/autre si tu veux le partager]
- **Jeu de données** — [À COMPLÉTER — nom du dataset, source, lien de téléchargement, ex. Kaggle "Chest X-Ray Images", "Brain Tumor MRI Dataset"].
- **Modèles entraînés (`.keras`)** — trop volumineux pour Git. [À COMPLÉTER — lien de téléchargement, ex. Google Drive / Hugging Face Hub].

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Placer les modèles téléchargés dans un dossier `models/` :
```
models/
├── type_image_classification_model.keras
├── poumons_classification_model.keras
├── cerveau_classification_model.keras
└── type_image_classes.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

## Avertissement

Cet outil est une aide à la décision fondée sur l'intelligence artificielle. Il ne remplace en aucun cas l'avis d'un professionnel de santé.
