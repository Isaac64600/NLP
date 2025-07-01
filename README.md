# NLP

## 📰 BBC News Article Classification & Topic Modeling (ML/DL + LDA)
### 📚 Description
Ce projet NLP (Natural Language Processing) a pour objectif de :

- Classer automatiquement des articles de presse de la BBC en différentes catégories (e.g. business, sport, tech...) en utilisant à la fois :

  - des modèles de Machine Learning (ML) avec vectorisation de texte ;

   - des modèles de Deep Learning (DL) avec Word2Vec et embeddings ;

- Extraire des mots-clés thématiques à partir d’un large corpus de texte (20 Newsgroups) via LDA (Latent Dirichlet Allocation).

### 🗂️ Données
- BBC Dataset : bbc_news_text_complexity_summarization.csv, contenant des articles et leurs catégories.


- 20 Newsgroups : Utilisé pour l’extraction de mots-clés thématiques via LDA.

### 🧼 Prétraitement
Deux types de nettoyage sont appliqués aux textes :

- clean_text_ML : sans suppression des stopwords (utile pour les modèles ML) ;

- clean_text_DL : avec suppression des stopwords (meilleur pour les embeddings DL).

### 🧠 Modèle Machine Learning
1.  Vectorisation du texte (TF-IDF).


2. Classification avec un modèle sélectionné automatiquement via select_classifier() (e.g. Logistic Regression, SVM).


3. Évaluation :
   - Validation croisée.
   - Courbes d’apprentissage pour precision_macro, recall_macro, et f1_macro.
   - Matrice de confusion.

### 🤖 Modèle Deep Learning
1. Tokenisation avec Keras + padding ;

2. Embeddings Word2Vec entraînés sur les textes BBC nettoyés ;

   3. Modèle DL personnalisé (type LSTM) avec :

        - Entrées : vecteurs Word2Vec.

        - Évaluation : précision sur le test set, matrice de confusion.

### 🧠 LDA - Extraction de Thèmes
Corpus : Dataset 20 Newsgroups nettoyé et tokenisé.

Modèle LDA (20 topics) entraîné avec gensim.

Affichage des 5 mots les plus représentatifs de chaque thème.

### 💾 Sauvegarde des Modèles
Tous les objets nécessaires à l’inférence sont sauvegardés :

vectorizer, tokenizer, ml_model, dl_model, lda_model, dictionary.

### 📊 Résultats
- ML Accuracy et métriques via validation croisée ;

- DL Accuracy sur le test set ;

- Topics LDA listés avec leurs mots-clés.

🛠️ Dépendances principales

```bash 
pandas
scikit-learn
gensim
tensorflow / keras
matplotlib
seaborn
nltk / spacy
```
