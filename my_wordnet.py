from nltk.corpus import wordnet as wn

print(wn.synsets('dog'))

print(wn.synsets('casa', lang='spa'))

for synset in wn.synsets('important').lemma_names('spa'):
    print(synset.)