from tkinter import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Natural_Language_Processing import le ,preprocessing
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from keras.models import load_model
import numpy as np
import pickle
import os


def load_models(path_prefix="models/"):
    with open(os.path.join(path_prefix, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(path_prefix, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(path_prefix, "ml_model.pkl"), "rb") as f:
        ml_model = pickle.load(f)

    dl_model = load_model(os.path.join(path_prefix, "dl_model.keras"))

    lda_model = LdaModel.load(os.path.join(path_prefix, "lda_model"))
    dictionary = Dictionary.load(os.path.join(path_prefix, "lda_dictionary.dict"))

    print("✅ All models including LDA loaded successfully.")
    return vectorizer, tokenizer, ml_model, dl_model, lda_model, dictionary


vectorizer, tokenizer, ml_model, dl_model, lda_model, dictionary = load_models()
root = Tk()
root.title("Text Classifier & Summarizer")
root.resizable(True, True)


BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


def classify(text, mode):


    if mode == "ML":
        #Nettoyage du texte (stopwords non supprimés)
        cleaned_text = preprocessing(text, remove_stopwords=False)
        #MODELE ML : transformation du texte en vecteur via le vectorizer entraîné
        vec = vectorizer.transform([cleaned_text])
        #Prédiction de la classe (valeur numérique) avec le modèle ML
        pred = ml_model.predict(vec)[0]

    elif mode == "DL":
        #Nettoyage du texte (stopwordssupprimés)
        cleaned_text = preprocessing(text, remove_stopwords=False)
        #MODELE DL : transformation du texte en séquences de tokens
        seq = tokenizer.texts_to_sequences([cleaned_text])
        #Padding des séquences pour uniformiser la taille
        padded = pad_sequences(seq, maxlen=100, padding="post")
        #Prédiction avec le modèle DL (argmax pour récupérer l'indice de la classe prédite)
        pred = np.argmax(dl_model.predict(padded), axis=1)[0]

    else:
        #Gestion des cas où le mode est incorrect
        return "Invalid mode selected."

    #Récupération des noms de classe d'origine (e.g. "sport", "tech", etc.)
    class_names = le.classes_

    #Retour de la prédiction au format lisible
    return f"Predicted class: {class_names[pred]}"

def lda_summary(text, top_n=2):
    #Nettoyage et tokenisation du texte
    tokens = preprocessing(text).split()
    #Transformation du texte en sac de mots (BoW) basé sur le dictionnaire LDA
    bow = dictionary.doc2bow(tokens)

    #Si aucun mot n'est reconnu par le modèle, on retourne un message d'erreur
    if not bow:
        return "Topic keywords: (input too different from training corpus)"

    #Récupération des topics associés au document avec leur probabilité
    topics = lda_model.get_document_topics(bow)
    #Sélection des top_n topics les plus dominants (triés par poids décroissant)
    dominant_topics = sorted(topics, key=lambda x: -x[1])[:top_n]

    keywords = []
    #Pour chaque topic dominant, récupérer les top mots-clés associés
    for topic_num, _ in dominant_topics:
        topic_terms = lda_model.show_topic(topic_num, topn=5)
        keywords.extend([term for term, _ in topic_terms])

    # onstruction du résumé textuel des mots-clés (sans doublons)
    summary = "Topic keywords: " + ", ".join(set(keywords))
    return summary


def send():
    user_input = e.get()
    mode = model_choice.get()

    txt.insert(END, f"\nYou -> {user_input}")

    if mode not in ["ML", "DL"]:
        response = "Please select a classification model (ML or DL)."
    else:
        class_result = classify(user_input, mode)
        summary_result = lda_summary(user_input)
        response = f"{class_result}\n{summary_result}"

    txt.insert(END, f"\nBot -> {response}")
    e.delete(0, END)


Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="NLP Bot", font=FONT_BOLD, pady=10)\
    .grid(row=0, column=0, columnspan=2, sticky="ew")


frame = Frame(root)
frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

txt = Text(frame, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, wrap=WORD)
txt.grid(row=0, column=0, sticky="nsew")

scrollbar = Scrollbar(frame, command=txt.yview)
scrollbar.grid(row=0, column=1, sticky='ns')
txt.config(yscrollcommand=scrollbar.set)


e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
e.grid(row=2, column=0, sticky="ew", padx=2, pady=2)

send_btn = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY, command=send)
send_btn.grid(row=2, column=1, sticky="ew", padx=2, pady=2)


model_choice = StringVar(root)
model_choice.set("ML")
option_menu = OptionMenu(root, model_choice, "ML", "DL")
option_menu.config(font=FONT_BOLD, bg=BG_GRAY)
option_menu.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=4)

root.mainloop()