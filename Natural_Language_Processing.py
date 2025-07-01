from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
import nltk
import re
from num2words import num2words
from nltk.corpus import wordnet
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from gensim.models import CoherenceModel, Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
from keras import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_validate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Bidirectional
from gensim.models import LdaModel
import pickle
from sklearn.datasets import fetch_20newsgroups

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

#Nettoyer et Normaliser la donnees
def preprocessing(sentence,remove_stopwords=False):
    #Initialiser le stop_words set
    stop_words = set(stopwords.words('english'))
    #Supprime les espaces blancs de début et de fin.
    sentence = sentence.strip()
    #Convertit tout le texte en minuscules
    sentence = sentence.lower()
    #Remplace les chiffres par leurs équivalents en mots
    sentence = re.sub(r'\d+', lambda match: num2words(match.group()), sentence)
    #Suppression de la ponctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')
    #Initialisation de la tokenisation et du lemmatiseur
    lemmatizer = WordNetLemmatizer()
    #Casse la phrase en plusieurs mots
    tokens = word_tokenize(sentence)
    #Suppression des mots d'arrêt "optionnel"
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    #Etiquette chaque mots avec son role grammatical
    pos_tagged = pos_tag(tokens)
    result = []
    for word, tag in pos_tagged:
        if tag.startswith('VB'):
            tag = wordnet.VERB
        elif tag.startswith('NN'):
            tag = wordnet.NOUN
        elif tag.startswith('JJ'):
            tag = wordnet.ADJ
        else:
            tag = wordnet.NOUN
        #Chaque mot est lemmatisé en fonction de son POS tag.
        lemmatized_word = lemmatizer.lemmatize(word, pos=tag)
        result.append(lemmatized_word)
    #Les Tokens sont réunis en une seule chaîne de caractères séparée par des espaces.
    cleaned_sentence = ' '.join(result)

    return cleaned_sentence

#Préparer des données d'entraînement et de test pour le deep learning
def dl_preprocessing(X_train, X_test):
    #Intialisation du Tokenizer
    tokenizer = Tokenizer()
    #Apprend la fréquence des mots et construit le dictionnaire "word_index"
    tokenizer.fit_on_texts(X_train)
    #Converti le text en une liste de INT (ou chaque int est un token id qui represente un mot)
    X_train_token = tokenizer.texts_to_sequences(X_train)
    X_test_token = tokenizer.texts_to_sequences(X_test)
    #Padding
    X_train_padded = pad_sequences(X_train_token, dtype="int32", padding="post", maxlen=100)
    X_test_padded = pad_sequences(X_test_token, dtype="int32", padding="post", maxlen=100)
    return X_train_padded, X_test_padded, tokenizer

#Cette fonction entraîne un modèle de classification de texte à l'aide d'un SVM ou d'une régression logistique, en fonction de la taille du dataset.
def select_classifier(X_train, y_train,X_test,y_test, threshold=10000):
    if len(X_train) >= threshold:
        print(f"Using SVM (dataset size: {len(X_train)} ≥ {threshold})")
        model = svm.SVC(kernel='linear', probability=True)  # use linear kernel for text
    else:
        print(f"Using Logistic Regression (dataset size: {len(X_train)} < {threshold})")
        model = LogisticRegression(max_iter=1000)
    #Initialisation du Vectorizer
    vectorizer = TfidfVectorizer()
    #Converti le texte en vecteur
    X_train_vec = vectorizer.fit_transform(X_train)
    #Veille à ce que les données de test soient vectorisées à l'aide du même vocabulaire.
    X_test_vec = vectorizer.transform(X_test)
    #Entrainement du model et evaluation
    model.fit(X_train_vec, y_train)
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    print("train_score=", train_score)
    print("test_score=", test_score)
    return model, model.__class__.__name__,vectorizer

def LDA(docs,num_topics, threshold=10000):
    #Crée un mapping ID ↔ mot à partir des textes tokenisés
    dictionary = Dictionary(docs)
    #Nettoie le vocabulaire pour enlever mots trop rares ou trop fréquents
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=threshold)
    #Convertit chaque document en représentation bag of words
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    #Entrainement du LDA
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100)
    #Cohérence du modèle
    coherence_model_lda = CoherenceModel(model=lda, texts=docs, dictionary=dictionary,processes=1)
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)
    return lda, corpus, dictionary

#Intialisation et entrainement du model de deep learning
def dl_model(X_train, y_train, word_index, word2vec_model, input_length=100,
             lstm_units=128, dense_units=64, dropout_rate=0.3, batch_size=64, epochs=20):
    #Calcul de la taille des vecteurs de mots
    embedding_dim = word2vec_model.vector_size
    #Créer une matrice de poids pour la couche d'embedding
    embedding_matrix = build_embedding_matrix(word_index, word2vec_model, embedding_dim)
    #Calcul de la taille du vocabulaire
    vocab_size = len(word_index) + 1
    #Calcul du nombre de classes
    num_classes = len(np.unique(y_train))
    #Intialisation du Model
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=input_length,
                  trainable=True),
        #Capte les dépendances avant et après un mot dans la phrase.
        Bidirectional(LSTM(units=lstm_units, return_sequences=False)),
        #Régularisation pour éviter l’overfitting.
        Dropout(dropout_rate),
        Dense(units=dense_units, activation="relu"),
        Dropout(dropout_rate / 1.5),
        Dense(units=num_classes, activation="softmax")
    ])
    #Compilation du modèle
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    #Arrêt anticipé si la perte de validation ne s’améliore pas pendant 3 epochs.
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #Réduit le taux d’apprentissage si la validation stagne.
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    #Entraînement du modèle
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        #Callbacks pour gestion dynamique
        callbacks=[early_stop, lr_schedule]
    )

    return model

#Cree une matrice ou chaque ligne represente un vecteur de mots Word2Vec
def build_embedding_matrix(word_index, word2vec_model, embedding_dim):
    #Calcul de la taille du vocabulaire
    vocab_size = len(word_index) + 1
    #Initialisation de la Matrice d'embedding
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    #Remplir la matrice avec des vecteur de mots
    for word, i in word_index.items():
        #Si le mot est present dans le model word2vec pré-entraînée
        if word in word2vec_model.wv:
            #Associe chaque mot à son vecteur Word2Vec
            embedding_matrix[i] = word2vec_model.wv[word]
        else:
            #Remplir par un vecteur random en utilisant une distribution normale avec un écart-type modéré (0,6) afin d'imiter la distribution d'embedding pré-entraînée.
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return embedding_matrix
#Cette fonction applique une validation croisée à 5 plis sur des données textuelles déjà vectorisables, en évaluant plusieurs métriques de classification.
def cross_validation(X_train, y_train, model,vectorizer):
    #Metric d'evaluation
    scoring = ['precision_macro', 'recall_macro',"f1_macro"]
    #Vectorisation du texte
    X_train_vec = vectorizer.transform(X_train)
    #Validation croisée
    cv_scores = cross_validate(model, X_train_vec, y_train, cv=5, scoring=scoring)
    #Affichage des résultats
    print("Cross-validation results:")
    for metric in scoring:
        mean_score = np.mean(cv_scores[f'test_{metric}'])
        print(f"{metric}: {mean_score:.4f}")


def plot_learning_curve(model, X, y, title="Learning Curve", cv=5, scoring='accuracy'):

    pipeline = make_pipeline(TfidfVectorizer(), model)
    #Génération des courbes d’apprentissage
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, shuffle=True, random_state=42
    )
    #Moyennes des scores
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    #Affichage du graphique
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', label="Validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring.capitalize())
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Permet d’afficher une matrice de confusion pour évaluer les performances d’un classifieur, qu’il soit basé sur du machine learning classique ou du deep learning.
def confusion_matrix_display(X_test, model, y_test, vectorizer=None, label_encoder=None):
    #Si un vectorizer est fourni
    if vectorizer is not None:
        #On transforme les textes bruts en vecteurs
        X_test_vec = vectorizer.transform(X_test)
        pred = model.predict(X_test_vec)
        display_labels = model.classes_
    else:
        pred_probs = model.predict(X_test)
        pred = np.argmax(pred_probs, axis=1)
        #Pour savoir quelles sont les classes dans le bon ordre, on utilise label_encoder.classes_ si fourni.Sinon, on utilise np.unique(y_test).
        display_labels = label_encoder.classes_ if label_encoder else np.unique(y_test)
    #Création de la matrice de confusion
    cm = confusion_matrix(y_test, pred, labels=range(len(display_labels)))
    #Affichage du graphique
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


def save_models(vectorizer=None, tokenizer=None, ml_model=None, dl_model=None, lda_model=None, dictionary=None, path_prefix="models/"):
    os.makedirs(path_prefix, exist_ok=True)
    #Sauvegarder le vectorizer
    if vectorizer is not None:
        with open(os.path.join(path_prefix, "vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
    #Sauvegarder le tokenizer
    if tokenizer is not None:
        with open(os.path.join(path_prefix, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
    #Sauvegarder le model ML
    if ml_model is not None:
        with open(os.path.join(path_prefix, "ml_model.pkl"), "wb") as f:
            pickle.dump(ml_model, f)
    #Sauvegarder le model DL
    if dl_model is not None:
        dl_model.save(os.path.join(path_prefix, "dl_model.keras"))
    #Sauvegarder le model LDA
    if lda_model is not None:
        lda_model.save(os.path.join(path_prefix, "lda_model"))
    #Sauvegarder le dictionnaire
    dictionary.save(os.path.join(path_prefix, "lda_dictionary.dict"))
    print("✅ All models saved successfully.")

#Charger les donnees
df = pd.read_csv('bbc_news_text_complexity_summarization.csv')

df['clean_text_ML'] = df['text'].apply(lambda x: preprocessing(x, remove_stopwords=False))
df['clean_text_DL'] = df['text'].apply(lambda x: preprocessing(x, remove_stopwords=True))
#Initialisation du LabelEncoder
le = LabelEncoder()
##Labelisation des étiquettes
df["labels_encoded"] =  le.fit_transform(df.labels)
#Séparation des données (ML) en ensembles d'entraînement et de test (80/20)
X_train_ML, X_test_ML, y_train_ML, y_test_ML = train_test_split(df.clean_text_ML,df.labels_encoded, test_size=0.2, random_state=42)
#Séparation des données (DL) en ensembles d'entraînement et de test (80/20)
X_train_DL, X_test_DL, y_train_DL, y_test_DL = train_test_split(df.clean_text_DL,df.labels_encoded, test_size=0.2, random_state=42)
#Sélection et entraînement du meilleur modèle ML + vectorisation du texte
ml_model,name,vectorizer = select_classifier(X_train_ML, y_train_ML,X_test_ML, y_test_ML)
#Évaluation du modèle ML par validation croisée
cross_validation(X_train_ML, y_train_ML, ml_model,vectorizer)
#Tracer la courbe d'apprentissage avec scoring
plot_learning_curve(ml_model, X_train_ML, y_train_ML, title=f"Learning Curve (precision_macro): {name}", scoring='precision_macro')
plot_learning_curve(ml_model, X_train_ML, y_train_ML, title=f"Learning Curve (recall_macro): {name}", scoring="recall_macro")
plot_learning_curve(ml_model, X_train_ML, y_train_ML, title=f"Learning Curve (f1_macro): {name}", scoring="f1_macro")
#Affichage de la matrice de confusion pour le modèle ML sur l'ensemble de test
confusion_matrix_display(X_test=X_test_ML,vectorizer=vectorizer,model =ml_model,y_test=y_test_ML)
#Tokenisation des textes nettoyés (DL) pour entraînement de Word2Vec
tokenized_texts = [text.split() for text in df['clean_text_DL']]
#Entraînement du modèle Word2Vec sur les textes BBC nettoyés/tokenisés
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1)
#Prétraitement des textes pour le modèle DL : tokenisation + padding
X_train_padded, X_test_padded, tokenizer= dl_preprocessing(X_train_DL, X_test_DL)
#Entraînement du modèle DL avec embeddings Word2Vec
dl_model = dl_model(X_train_padded, y_train_DL, tokenizer.word_index,w2v_model)
#Évaluation du modèle DL sur l’ensemble de test
test_loss, test_acc = dl_model.evaluate(X_test_padded, y_test_DL)
print(f"Test Accuracy: {test_acc:.4f}")
#Affichage de la matrice de confusion du modèle DL
confusion_matrix_display(X_test=X_test_padded,y_test=y_test_DL,model=dl_model,label_encoder=le)
#Chargement du dataset 20 Newsgroups pour LDA
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
#Nettoyage et tokenisation des documents pour LDA
tokenized_docs = [preprocessing(doc, remove_stopwords=True).split() for doc in documents]
#Entraînement du modèle LDA avec 20 topics
lda_model, corpus, dictionary = LDA(tokenized_docs, num_topics=20)
#Sauvegarde de tous les modèles et objets utiles pour l’inférence
save_models(vectorizer=vectorizer, tokenizer=tokenizer,lda_model=lda_model, dictionary=dictionary,ml_model= ml_model, dl_model=dl_model)
#Affichage des 5 mots-clés les plus représentatifs pour chaque topic LDA
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx}: {topic}")
