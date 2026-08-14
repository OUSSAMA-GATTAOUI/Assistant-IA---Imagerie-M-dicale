# Assistant IA — Imagerie Médicale (Radiographie Thoracique & IRM Cérébrale)

Système d'intelligence artificielle d'aide au diagnostic, développé dans le cadre d'un stage d'initiation au **service informatique du Centre Hospitalier Universitaire Ibn Sina de Rabat**, sous l'encadrement du **Dr Yassine Sembati**.

Le système combine trois réseaux de neurones convolutifs (DenseNet121, apprentissage par transfert) organisés en cascade : un modèle "aiguilleur" identifie automatiquement le type d'image soumise (radiographie thoracique / IRM cérébrale / autre), puis route l'image vers le modèle spécialiste correspondant. Les prédictions sont accompagnées de cartes d'explicabilité **Grad-CAM** et restituées via une application web **Streamlit** générant un compte-rendu structuré exportable en PDF.

## Architecture du système

```
Image soumise
      │
      ▼
Modèle Aiguilleur (3 classes : Radiographie / IRM / Autre)
      │
      ├── Radiographie thoracique ──▶ Modèle Pulmonaire (COVID-19, Opacité pulmonaire, Normal, Pneumonie virale)
      ├── IRM cérébrale ──────────────▶ Modèle Cérébral (Gliome, Méningiome, Tumeur hypophysaire, Sans tumeur)
      └── Autre / confiance faible ──▶ Image rejetée, message explicatif à l'utilisateur
                                              │
                                              ▼
                                  Grad-CAM + Compte-rendu structuré (TXT / PDF)
```

Chaque modèle spécialiste repose sur DenseNet121 pré-entraîné sur ImageNet (couches convolutives gelées), suivi d'un bloc de couches denses (1024 → 512 → 256 neurones, ReLU, BatchNorm, Dropout 40 %) et d'une couche de sortie softmax.

## Résultats

### Performance sur jeux de test internes

| Modèle | Exactitude | Précision moyenne | Rappel moyen |
|---|---|---|---|
| Pulmonaire | 93,1 % | 93,2 % | 93,0 % |
| Cérébral | 98,0 % | 98,3 % | 97,9 % |
| Aiguilleur | 100 % | 100 % | 100 % |

### Validation externe (jeux de données indépendants)

| Modèle | Résultat externe |
|---|---|
| Pulmonaire | 92,2 % d'exactitude globale, mais seulement 81,2 % sur la classe COVID-19 (contre >96 % pour Normal/Pneumonie) |
| Cérébral | 100 % de détection des cas "sans tumeur", 87 % de détection des cas tumoraux (toutes catégories confondues) |

La validation externe révèle une baisse de fiabilité par rapport aux jeux de test internes, notamment sur la classe COVID-19. Cet écart met en évidence un risque d'**apprentissage de raccourcis** (le modèle apprenant des caractéristiques propres à sa source de données plutôt que des signes radiologiques universellement transférables) plutôt qu'un défaut d'exactitude en tant que tel — voir la section *Limites* ci-dessous et le rapport complet pour l'analyse détaillée.

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Langage | Python 3.12 |
| Deep learning | TensorFlow / Keras |
| Architecture CNN | DenseNet121 (apprentissage par transfert depuis ImageNet) |
| Explicabilité | Grad-CAM |
| Interface web | Streamlit |
| Génération de rapport PDF | ReportLab |
| Traitement d'image | Pillow, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Métriques / préparation des données | scikit-learn, split-folders |

## Jeux de données utilisés

- **COVID-19 Radiography Database** (Chowdhury et al., Rahman et al.) — radiographies thoraciques réparties en COVID-19, opacité pulmonaire, normal, pneumonie virale.
- **Brain Tumor MRI Dataset** (M. Nickparvar, Kaggle) — IRM cérébrales réparties en gliome, méningiome, tumeur hypophysaire, sans tumeur.
- **Jeu de données de l'aiguilleur** — combinaison d'échantillons des deux jeux ci-dessus avec des images CIFAR-10 et un jeu de tomodensitométrie thoracique complémentaire, pour entraîner la distinction radiographie / IRM / autre.

Les jeux de données bruts et prétraités **ne sont pas inclus dans ce dépôt** (volume, licences). Se référer aux sources citées ci-dessus pour les télécharger.

## Contenu de ce dépôt

Ce dépôt contient uniquement :
- `app.py` — l'application Streamlit complète (chargement des modèles, contrôle qualité, inférence, Grad-CAM, génération du compte-rendu PDF).
- le notebook d'entraînement des modèles.
- `requirements.txt`.

**Ne sont pas inclus** (voir rapport complet pour le détail) :
- le notebook de validation externe,
- les jeux de données (bruts et prétraités),
- les modèles entraînés au format `.keras` (trop volumineux — à héberger séparément, ex. Google Drive / Hugging Face Hub).

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Placer les modèles entraînés dans un dossier `models/` à la racine du projet :

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

L'interface s'ouvre par défaut sur `http://localhost:8501`.

## Limites connues

- **Généralisation imparfaite** : la validation externe montre une baisse de fiabilité par rapport aux performances internes, en particulier sur la classe COVID-19 (81,2 % contre >96 % pour les autres classes pulmonaires), suggérant un possible apprentissage de caractéristiques propres à la source d'acquisition plutôt que de signes radiologiques réellement transférables.
- **Pas de persistance des données** : l'historique des analyses est conservé en mémoire de session Streamlit uniquement (pas de base de données), ce qui limite le suivi longitudinal et l'usage multi-utilisateurs.
- **Environnement d'entraînement hétérogène** : une partie de l'entraînement a été réalisée en local (CPU, sans GPU) et une autre sur Google Colab, ce qui a introduit une hétérogénéité de structure entre les notebooks.
- **Classes ambiguës** : les confusions les plus fréquentes du modèle pulmonaire se situent entre "opacité pulmonaire" et "normal", des catégories dont les frontières radiologiques ne sont pas toujours nettement tranchées, y compris pour un lecteur humain.

## Avertissement

Cet outil est un **prototype d'aide à la décision** fondé sur l'intelligence artificielle, développé dans un cadre académique (stage d'initiation). Il ne remplace en aucun cas l'avis d'un professionnel de santé et ne doit pas être utilisé pour un diagnostic clinique réel sans validation médicale et réglementaire appropriée.

## Remerciements

Ce travail a été réalisé sous l'encadrement du **Dr Yassine Sembati**, au sein du service informatique du **Centre Hospitalier Universitaire Ibn Sina de Rabat**.
