import matplotlib.pyplot as plt
import pickle
import os


def plot_viceversa(word, palabra, en_sim, es_sim):
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(4, 2))

    # Horizontal bar for value1 (left side)
    ax.barh(y=0.1, width=en_sim, height=0.4, color='blue', label=word, align='center')

    # Horizontal bar for value2 (right side)
    ax.barh(y=-0.1, width=es_sim, height=0.4, color='red', label=palabra, align='center', left=100-es_sim)

   # Add labels and formatting
    ax.set_xlim(0, 100)  # Scale from 0 to 100
    ax.set_yticks([])    # Remove y-axis ticks

    # Add labels directly on the sides
    ax.text(-3, 0.1, word, va='center', ha='right', fontsize=22, color='black')  # Left label
    ax.text(103, -0.1, palabra, va='center', ha='left', fontsize=22, color='black')

    ax.set_xlabel("Similarity (%)")

    out_path = os.path.join("graphics", f"bar_{word}_{palabra}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

with open("wordpala_results.pickle", "rb") as f:
    results = pickle.load(f)

results.apply(lambda row: plot_viceversa(row['original_word'], row['translated_word'], row['en_sim'], row['es_sim']), axis=1)
