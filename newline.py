import pandas as pd
import spacy
from deep_translator import GoogleTranslator
import random
import string
import pickle
import nltk
from nltk.corpus import wordnet as wn

DEBUG = False

def preprocess(filename):
    # Load the source and target sentences
    with open("tatoeba_lg/eng.txt", "r", encoding="utf-8") as src_file:
        source_sentences = src_file.read().splitlines()

    with open("tatoeba_lg/esp.txt", "r", encoding="utf-8") as tgt_file:
        target_sentences = tgt_file.read().splitlines()

    # Check if both files have the same number of lines
    if DEBUG and len(source_sentences) != len(target_sentences):
        print(f"englen: {len(source_sentences)} :: esplen: {len(target_sentences)}")
        print("Warning: The source and target files have different numbers of sentences.")

    # Create a DataFrame with the sentence pairs
    df = pd.DataFrame({'en': source_sentences, 'es': target_sentences})

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

    # Apply the function to the DataFrame columns
    df['en_tokens'] = df['en'].apply(lambda x: tokenize_and_lemmatize(x, lang='en'))
    df['es_tokens'] = df['es'].apply(lambda x: tokenize_and_lemmatize(x, lang='es'))

    # Display the DataFrame with tokens
    print(df[['en', 'en_tokens', 'es', 'es_tokens']])
    df.to_csv(filename, index=True)

def preprocess_to_pickle(pickle_file):
    # Load the source and target sentences
    with open("tatoeba_lg/eng.txt", "r", encoding="utf-8") as src_file:
        source_sentences = src_file.read().splitlines()
    with open("tatoeba_lg/esp.txt", "r", encoding="utf-8") as tgt_file:
        target_sentences = tgt_file.read().splitlines()

    # Check if both files have the same number of lines
    if DEBUG and len(source_sentences) != len(target_sentences):
        print(f"englen: {len(source_sentences)} :: esplen: {len(target_sentences)}")
        print("Warning: The source and target files have different numbers of sentences.")

    # Create a DataFrame with the sentence pairs
    df = pd.DataFrame({'en': source_sentences, 'es': target_sentences})

    # Load spaCy language models
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_sm")

    def tokenize_lemmatize_pos(text, lang):
        if lang == 'en':
            doc = nlp_en(text)
        elif lang == 'es':
            doc = nlp_es(text)
        else:
            raise ValueError("Language not supported. Use 'en' or 'es'.")

        # Tokenize, lemmatize, and extract POS, ignoring punctuation
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        pos_tags = [token.pos_ for token in doc if not token.is_punct]

        return lemmas, pos_tags


    # Apply the function to the DataFrame columns
    df[['en_lemmas', 'en_poses']] = df['en'].apply(lambda x: pd.Series(tokenize_lemmatize_pos(x, lang='en')))
    df[['es_lemmas', 'es_poses']] = df['es'].apply(lambda x: pd.Series(tokenize_lemmatize_pos(x, lang='es')))

    # Serialize the DataFrame using pickle
    with open(pickle_file, 'wb') as pf:
        pickle.dump(df, pf)

    print("Preprocessing complete. Data saved to Pickle file.")

# Function to translate a word
def translate_word(word, source_lang='en', target_lang='es'):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(word)

# Function to calculate the percentage of translated sentences containing the word
def get_sim(lang, source_word, target_word, df):
    # Ensure the value is a string
    if not isinstance(source_word, str) or not isinstance(target_word, str):
        raise ValueError("The words must be strings.")
    if not isinstance(lang, str):
        raise ValueError("The lang must be a string.")
    
    source = lang + "_lemmas"

    if lang == "en":
        otro = "es"
    elif lang == "es":
        otro = "en"
    else:
        otro = "UNK"
    target = otro + "_lemmas"

    return get_sim_st(source, target, source_word, target_word, df)

def get_sim_st(source, target, source_word, target_word, df):
    ## TODO: añadir código para ignorar los casos en los que el POS de source_word es diferente de target_word

    def contains_exact_word(tokens, target_word):
        # print("tokens are ", tokens, ", tw is ", target_word)
        for token in tokens:
            if token == target_word:
                # print(f"{target_word} in {tokens}")
                return True
        # print(f"{target_word} is NOT in {tokens}")
        return False

    filtered_df = df[df[source].apply(
        lambda text: contains_exact_word([token.lower() for token in text], source_word.lower())
    )]
    
    num_sentences = len(filtered_df)
    if num_sentences == 0:
        if DEBUG:
            print(f"Warning no sentences were found with {source_word}")
        return float("NaN")

    count = filtered_df[target].apply(        
        lambda text: contains_exact_word([token.lower() for token in text], target_word.lower())
        ).sum()

    # Calculate percentage
    return (count / num_sentences) * 100

## random sampling
# Randomly select 100 rows
# sampled_df = df.sample(n=100, random_state=1)

# Step 2: Define a function to select one random word from a sentence
def get_random_word(sentence):
    # Filter out punctuation from the tokenized sentence
    filtered_words = [word for word in sentence if word not in string.punctuation]
    
    # Check if there are any valid words left after filtering
    if filtered_words:
        return random.choice(filtered_words)
    else:
        return None  # Return None if no valid words are found

# # Step 3: Apply the function to each row
# sampled_df['random_word_en'] = sampled_df['en_tokens'].apply(get_random_word)
# sampled_df['random_word_es'] = sampled_df['es_tokens'].apply(get_random_word)

# # Iterate over random words to translate and calculate percentage
# results = []

# for index, row in sampled_df.iterrows():
#     word = row['random_word_en']
#     assert isinstance(word, str)
#     assert len(word) < 500

#     palabra = translate_word(word, source_lang='en', target_lang='es')
#     word = str(word)
#     palabra = str(palabra)

#     # Calculate the percentage of sentences in the entire DataFrame that include the translated word
#     es_over_total_filtered = percentage_es_over_total_filtered(df, word, palabra)
#     es_over_total_es = get_es_sim(df, word, palabra)
    
#     # Append results
#     results.append({
#         'original_word': word,
#         'translated_word': palabra,
#         'es_over_total_filtered': es_over_total_filtered,
#         'es_over_total_es': es_over_total_es
#     })

# # Convert results to DataFrame for easier viewing
# results_df = pd.DataFrame(results)

def disp_total(df, cmd):
    if "r" in cmd:
        # Adjust display options to show all rows and columns
        pd.set_option('display.max_rows', None)  # None means no limit on rows
    if "c" in cmd:
        pd.set_option('display.max_columns', None)  # None means no limit on columns

    # Print the whole DataFrame
    print(df)

    if "r" in cmd:
        pd.reset_option('display.max_rows')
    if "c" in cmd:
        pd.reset_option('display.max_columns')

def disp_results(results_df):
    # Adjust display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # None means no limit on rows
    pd.set_option('display.max_columns', None)  # None means no limit on columns

    # Print the whole DataFrame
    print(results_df)

    # Optionally, reset the options back to default after printing
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

    print(f"Avg. English similarity: {results_df['en_sim'].mean()}")
    print(f"Avg. Spanish similarity: {results_df['es_sim'].mean()}")

# display_results(results_df)
# results_df.to_csv('100_random.csv', index=True)

# Iterate over random words to translate and calculate percentage

# preprocess("lemmatized_1.csv")
# print("preprocess done")

# def random_

# this is for old nltk method
def en_to_es(word, pos="UNK"):
    if pos == "NOUN":
        syns = wn.synsets(word, pos=wn.NOUN)
    elif pos == "VERB":
        syns = wn.synsets(word, pos=wn.VERB)
    else:
        syns = wn.synsets(word)

    if syns:
        return [lemma.name() for lemma in syns[0].lemmas('spa')]  # Traducción al español
    else:
        return None

# this is for old nltk method
def create_en_es_dataframe_nltk(df):
    df["es"] = df.apply(lambda row: en_to_es(row["en"], row["en_pos"]), axis=1)
    return df

def translate_dataframe(df):
    '''
    input: dataframe with a column 'en'
    output: dataframe with new column of translations to es
    '''
    translator = GoogleTranslator(source='en', target='es')

    # Aplicar el traductor a cada fila del DataFrame
    df['es'] = df['en'].apply(lambda x: translator.translate(x))

    return df

def get_results_translate(words, df, source_lang, target_lang):
    results = []

    for word in words:
        palabra = translate_word(word, source_lang, target_lang)
        
        # Append results
        results.append({
            'original_word': word,
            'translated_word': palabra,
            'en_sim': get_en_sim(df, word, palabra),
            'es_sim': get_es_sim(df, word, palabra),
            'lang': str(source_lang + " -> " + target_lang)
        })

    return pd.DataFrame(results)

'''Takes in a list of tuples (word, palabra)'''
def make_results(worpal, df):
    results = []

    for word, palabra in worpal:
        # Append results
        results.append({
            'original_word': word,
            'translated_word': palabra,
            'en_sim': get_sim("en", word, palabra, df),
            'es_sim': get_sim("es", palabra, word, df),
        })

    return pd.DataFrame(results)

def get_pos(word, nlp):
    doc = nlp(word)
    if DEBUG and len(doc) != 1:
        print(f"Warning: ignoring subsequent words in {word}")

    return doc[0].pos_

def results_with_pos(worpal, df):
    results = []

    en_nlp = spacy.load("en_core_web_sm")
    es_nlp = spacy.load("es_core_news_sm")

    for word, palabra in worpal:
        # Append results
        results.append({
            'original_word': word,
            'translated_word': palabra,
            'en_sim': get_sim("en", word, palabra, df),
            'en_pos': get_pos(word, en_nlp),
            'es_sim': get_sim("es", palabra, word, df),
            'es_pos': get_pos(palabra, es_nlp),
        })

    return pd.DataFrame(results)

def results_from_df(worpal_df, df):
    results = []

    en_nlp = spacy.load("en_core_web_sm")
    es_nlp = spacy.load("es_core_news_sm")

    for word, palabra in zip(worpal_df["en"], worpal_df["es"]):
        word = str(word)
        palabra = str(palabra)
        # Append results
        results.append({
            'original_word': word,
            'translated_word': palabra,
            'en_sim': get_sim("en", word, palabra, df),
            'en_pos': get_pos(word, en_nlp),
            'es_sim': get_sim("es", palabra, word, df),
            'es_pos': get_pos(palabra, es_nlp),
        })

    return pd.DataFrame(results)

def calc_pos_results(df):
    # Calcular promedios y conteos por categoría gramatical e idioma
    en_stats = df.groupby("en_pos").agg(
        en_avg=("en_sim", "mean"),
        en_count=("en_sim", "size")
    )

    es_stats = df.groupby("es_pos").agg(
        es_avg=("es_sim", "mean"),
        es_count=("es_sim", "size")
    )

    # Combinar resultados en un DataFrame
    stats_df = pd.concat([en_stats, es_stats], axis=1).fillna(0)

    # Mostrar resultados
    return stats_df

def make_worpal_df(df, filter_factor=1):
    just_en = df["en_lemmas"].explode().to_frame(name="en")
    just_en = just_en.drop_duplicates().reset_index(drop=True)

    # Filtrar cada filter_factor entrada
    just_en_filtered = just_en[just_en.index % filter_factor == 0]

    words_con_es_filtered = translate_dataframe(just_en_filtered)

    return words_con_es_filtered

def get_results(dataset_pickle, worpal_pickle, results_pickle, filter_factor):
    with open(dataset_pickle, "rb") as f:
        lemmatized = pickle.load(f)

    # pos-tagged and lemmatized dataset
    print(f"got lemmatized for {filter_factor}")

    # list of words and palabras that appear in dataset
    worpal_df = make_worpal_df(lemmatized, filter_factor)

    print(f"got worpal for {filter_factor}")
    with open(worpal_pickle, "wb") as f:
        pickle.dump(worpal_df, f)

    results_df = results_from_df(worpal_df, df=lemmatized)
    print(f"got results for {filter_factor}")
    print(results_df)
    with open(results_pickle, "wb") as f:
        pickle.dump(results_df, f)

def results_wordpala(lemmatized):
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
        ("indicate", "indicar"),
        ("encounter", "encontrar"),
        ("import", "importar"),
        ("signify", "significar"),
        
        # formality switch
        ("presume", "presumir"),
        ("create", "crear"),
        ("prepare", "preparar"),

        # pronouns
        ("you", "tú"),
        ("you", "vos"),
        ("you", "usted"),
        ("you", "vosotros"),
        ("you", "ustedes"),
        ("y'all", "ustedes"),
        ("yall", "ustedes"),

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

    return results_with_pos(wordpala, lemmatized)


if __name__ == "__main__":
    with open("lemma_pos.pickle", "rb") as f:
        lemmatized = pickle.load(f)

    my_results = results_wordpala(lemmatized)
    with open("wordpala_results.pickle", "wb") as f:
        pickle.dump(my_results, f)
    print(my_results)

    # with open("results_10.pickle", "rb") as f:
    #     resultos = pickle.load(f)
    # print(resultos)

