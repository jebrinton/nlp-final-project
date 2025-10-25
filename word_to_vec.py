import fasttext
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Load the pre-trained models
en_model = fasttext.load_model("cc.en.300.bin")
es_model = fasttext.load_model("cc.es.300.bin")

# Get word vectors
english_word_vector = en_model.get_word_vector("beautiful")  # Replace "hello" with your English word
spanish_word_vector = en_model.get_word_vector("pretty")   # Replace "hola" with your Spanish word

# Calculate cosine similarity
similarity = cosine_similarity(english_word_vector, spanish_word_vector)
print(f"Cosine similarity: {similarity}")




























wordpala = [
    # common "to do"-type
    # ("know", "saber"),
    # ("know", "conocer"),
    ("be", "ser"),
    ("be", "estar"),
    # ("do", "hacer"),
    # ("make", "hacer"),
    ("tomorrow", "mañana"),
    ("morning", "mañana"),
    ("morning", "madrugada"),
    # ("wait", "esperar"),
    # ("hope", "esperar"),

    # cognates
    # ("alcohol", "alcohol"),
    # ("sushi", "sushi"),
    ("popular", "popular"),
    # ("ideal", "ideal"),
    ("mountain", "montaña"),

    # formality switch
    ("anterior", "anterior"),
    ("prepare", "preparar"),

    # pronouns
    ("you", "tú"),
    ("you", "vos"),
    ("you", "usted"),
    ("you", "vosotros"),
    ("you", "ustedes"),

    # ("I", "yo"),
    ("we", "nosotros"),

    # everyday nouns
    ("eye", "ojo"),
    # ("stick", "palo"),
    ("hand", "mano"),
    ("desk", "escritorio"),
    ("winter", "invierno"),
    ("shoe", "zapato"),

    # everyday adjectives
    ("red", "rojo"),
    ("big", "grande"),
    ("interesting", "interesante"),
    ("important", "importante"),

    # regional words
    ("juice", "jugo"),
    ("juice", "zumo"),

    # ("cookie", "galleta"),
    # ("biscuit", "galleta"),

    # ("apartment", "apartamento"),
    # ("flat", "piso"),

    # unrelated words
    # ("word", "pala"),
    # ("at", "zapato"),
    ("think", "tiempo"),
    # ("computer", "árbol"),
    ("be", "decir"),
]