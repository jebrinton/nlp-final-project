import pandas as pd
from sacremoses import MosesTokenizer
import spacy
from deep_translator import GoogleTranslator
import random
import string

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

# # Translate English to Spanish
# translator = GoogleTranslator(source='en', target='es')
# result = translator.translate("to")
# print("English to Spanish:", result)

# # Translate Spanish to English
# translator = GoogleTranslator(source='es', target='en')
# result = translator.translate("para")
# print("Spanish to English:", result)

# eng_words = ["to", "for", "from", "of", "bike", "tree", "try", "the"]
# esp_words = ["a", "para", "de", "sobre", "bicicleta", "árbol", "intentar", "el"]

# def translate_to_sp(words):
#     translator = GoogleTranslator(source="en", target="es")
#     for word in words:
#         print(translator.translate(word))

# Function to translate a word
def translate_word(word, source_lang='en', target_lang='es'):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(word)

# Function to calculate the percentage of translated sentences containing the word
def calculate_percentage(df, word, palabra):
    # Ensure the value is a string
    if not isinstance(word, str) or not isinstance(palabra, str):
        raise ValueError("The words must be strings.")
    
    filtered_df = df[df['en_tokens'].apply(lambda tokens: word in tokens)]

    num_sentences = len(filtered_df)
    
    # Count sentences containing the translated word
    count = filtered_df['es_tokens'].apply(lambda tokens: palabra in tokens).sum()
    
    # Calculate percentage
    return (count / num_sentences) * 100

# Randomly select 100 rows
sampled_df = df.sample(n=100, random_state=1)

# Step 2: Define a function to select one random word from a sentence
def get_random_word(sentence):
    # Filter out punctuation from the tokenized sentence
    filtered_words = [word for word in sentence if word not in string.punctuation]
    
    # Check if there are any valid words left after filtering
    if filtered_words:
        return random.choice(filtered_words)
    else:
        return None  # Return None if no valid words are found

# Step 3: Apply the function to each row
sampled_df['random_word_en'] = sampled_df['en_tokens'].apply(get_random_word)
sampled_df['random_word_es'] = sampled_df['es_tokens'].apply(get_random_word)

# Iterate over random words to translate and calculate percentage
results = []

for index, row in sampled_df.iterrows():
    word = row['random_word_en']
    assert isinstance(word, str)
    assert len(word) < 500

    palabra = translate_word(word, source_lang='en', target_lang='es')
    word = str(word)
    palabra = str(palabra)

    # Calculate the percentage of sentences in the entire DataFrame that include the translated word
    percentage = calculate_percentage(df, word, palabra)
    
    # Append results
    results.append({
        'original_word': word,
        'translated_word': palabra,
        'percentage': percentage
    })

# Convert results to DataFrame for easier viewing
results_df = pd.DataFrame(results)

# Adjust display options to show all rows and columns
pd.set_option('display.max_rows', None)  # None means no limit on rows
pd.set_option('display.max_columns', None)  # None means no limit on columns

# Print the whole DataFrame
print(results_df)

# Optionally, reset the options back to default after printing
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

print(f"Avg. pct 'correct': {results_df['percentage'].mean()}")
