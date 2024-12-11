import spacy
import pandas as pd

# Load the source and target sentences
with open("tatoeba_sm/eng.txt", "r", encoding="utf-8") as src_file:
    source_sentences = src_file.read().splitlines()

with open("tatoeba_sm/esp.txt", "r", encoding="utf-8") as tgt_file:
    target_sentences = tgt_file.read().splitlines()

# Check if both files have the same number of lines
if len(source_sentences) != len(target_sentences):
    print(f"englen: {len(source_sentences)} :: esplen: {len(target_sentences)}")
    print("Warning: The source and target files have different numbers of sentences.")

# Create a DataFrame with the sentence pairs
df = pd.DataFrame({'eng': source_sentences, 'esp': target_sentences})

# Load spaCy language model
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# Function to tokenize and lemmatize
def tokenize_and_lemmatize(text, lang='en'):
    if lang == 'en':
        doc = nlp_en(text)
    elif lang == 'es':
        doc = nlp_es(text)
    else:
        raise ValueError("Language not supported. Use 'en' or 'es'.")
    
    # Tokenize and lemmatize, ignoring punctuation
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct]
    return lemmatized_words

'''
# Apply the function to the DataFrame columns
df['eng_tokens'] = df['eng'].apply(lambda x: tokenize_and_lemmatize(x, lang='en'))
df['esp_tokens'] = df['esp'].apply(lambda x: tokenize_and_lemmatize(x, lang='es'))

# Display the DataFrame with tokens
print(df[['eng', 'eng_tokens', 'esp', 'esp_tokens']])
'''


# nlp = nlp_es("Por qué estás aquí con las gafas?")

# for token in nlp:
#     # Print the token and its part-of-speech tag
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)

'''
load_model = spacy.load("en_core_web_lg")    # make sure to use larger package!

nlp1 = load_model("I like salty fries and hamburgers.")
nlp2 = load_model("Fast food tastes very good.")

# Similarity of two documents
print(nlp1, "<->", nlp2, nlp1.similarity(nlp2))

# Similarity of tokens and spans
french_fries = nlp1[2:4]
burgers = nlp1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))
'''

# load_mdl = spacy.load("xx_sent_ud_sm")

# nlp_en = load_mdl("This is a sentence")
# nlp_es1 = load_mdl("Esta es una oración")
# nlp_es2 = load_mdl("Esta es una oracion")
# nlp_es3 = load_mdl("Esta es una frase")

# from itertools import combinations

# variables = [nlp_en, nlp_es1, nlp_es2, nlp_es3]

# for (g, h) in combinations(variables, 2):
#     print(g, "<->", h, g.similarity(h))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

query_embedding = model.encode("This is a phrase")
passage_embeddings = model.encode([
    "Esta es una oración",
    "Esta es una oracion",
    "Esta es una frase",
])

similarity = model.similarity(query_embedding, passage_embeddings)
print("similarity 1: ", similarity)

query_embedding = model.encode("phrase")
passage_embeddings = model.encode([
    "oración",
    "oracion",
    "frase",
])

similarity = model.similarity(query_embedding, passage_embeddings)
print("similarity 2: ", similarity)

query_embedding = model.encode("apple")
passage_embeddings = model.encode([
    "apples",
    "naranja",
    "informar",
])

similarity = model.similarity(query_embedding, passage_embeddings)
print("similarity 3: ", similarity)

query_embedding = model.encode("beach")
passage_embeddings = model.encode([
    "playa",
    "oro",
    "montaña",
])

similarity = model.similarity(query_embedding, passage_embeddings)
print("similarity 4: ", similarity)

query_embedding = model.encode("We went to the beach together")
passage_embeddings = model.encode([
    "Fuimos a la playa",
    "Fuimos a la playa juntos",
    "Fuimos a la montaña",
])

similarity = model.similarity(query_embedding, passage_embeddings)
print("similarity 5: ", similarity)