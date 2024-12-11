import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_by_pos(df):
    plt.rcParams.update({'font.size': 14.5})
    # Crear un gráfico para en_pos
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="en_pos", y="en_sim", ci=None, palette="deep")
    plt.title("Similarity by Part of Speech (English)")
    plt.xlabel("Part of Speech")
    plt.ylabel("Similarity Score")
    
    out_path = os.path.join("graphics", f"en_pos.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    # Crear un gráfico para es_pos
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="es_pos", y="es_sim", ci=None, palette="deep")
    plt.title("Similarity by Part of Speech (Spanish)")
    plt.xlabel("Part of Speech")
    plt.ylabel("Similarity Score")

    out_path = os.path.join("graphics", f"es_pos.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

def main():
    '''
    'original_word': word,
    'translated_word': palabra,
    'en_sim': get_sim("en", word, palabra, df),
    'en_pos': get_pos(word, en_nlp),
    'es_sim': get_sim("es", palabra, word, df),
    'es_pos': get_pos(palabra, es_nlp),
    '''
    with open("results_10.pickle", "rb") as f:
        df = pickle.load(f)

    # drop POS with 0
    df = df[~df['en_pos'].isin(['SCONJ', 'PUNCT'])]
    df = df[~df['es_pos'].isin(['SCONJ', 'PUNCT'])]

    # print(df)

    # plot_by_pos(df)

    # Agrupar por categoría gramatical en inglés y calcular la media de las similitudes
    en_stats = df.groupby("en_pos")["en_sim"].mean()
    print(en_stats)

    # Agrupar por categoría gramatical en español y calcular la media de las similitudes
    es_stats = df.groupby("es_pos")["es_sim"].mean()
    print(es_stats)

    # df_sorted_en = df.sort_values(by="en_sim", ascending=False)
    # print(df_sorted_en)

    # df_sorted_es = df.sort_values(by="es_sim", ascending=False)
    # print(df_sorted_es)

    # with open("worpal_10.pickle", "rb") as f:
    #     dataset = pickle.load(f)

    # print(dataset)


if __name__ == "__main__":
    main()
