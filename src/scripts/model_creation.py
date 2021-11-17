import string

import fasttext
import nltk
import nltk.stem
import spacy
from gensim.models import Word2Vec
from spacy.lang.pl import Polish

nltk.download('punkt')
nltk.download('pl196x')
nltk.download('wordnet')
nltk.download('stopwords')


def create():
    with open("wikipedia_corpus.txt") as file:
        lines = file.readlines()
        whole = [line.rstrip() for line in lines]

    with open("data_tripadvisor_corpus.txt") as file:
        lines = file.readlines()
        train = [line.rstrip() for line in lines]

    model = fasttext.train_unsupervised('wikipedia_corpus.txt', verbose=True)
    model.save_model("models/model_wiki_base.bin")
    wiki_worlds = [word for sentance in whole[:int(len(whole) / 6)] for word in nltk.word_tokenize(sentance)]
    model = Word2Vec(sentences=wiki_worlds, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_wiki_base.model")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus.txt', model='skipgram', verbose=True)
    model.save_model("models/model_train_base.bin")
    train_worlds = [word for sentance in train for word in nltk.word_tokenize(sentance)]
    model = Word2Vec(sentences=train_worlds, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_train_base.model")

    stops = spacy.lang.pl.stop_words.STOP_WORDS
    s = nltk.stem.WordNetLemmatizer()

    strain_tokens = []
    for sentance in train:
        final_worlds = []
        worlds = nltk.word_tokenize(sentance)
        for world in worlds:
            if world not in stops and world not in string.punctuation:
                final_worlds.append(s.lemmatize(world).lower().strip())

        strain_tokens.append(final_worlds)

    swhole_tokens = []
    for sentance in whole[:int(len(whole) / 6)]:
        final_worlds = []
        worlds = nltk.word_tokenize(sentance)
        for world in worlds:
            if world not in stops and world not in string.punctuation:
                final_worlds.append(s.lemmatize(world).lower().strip())

        swhole_tokens.append(final_worlds)

    with open('wikipedia_corpus_clean.txt', 'a') as the_file:
        for sentence in swhole_tokens:
            the_file.write((" ".join(sentence) + "\n"))

    with open('data_tripadvisor_corpus_clean.txt', 'a') as the_file:
        for sentence in strain_tokens:
            the_file.write((" ".join(sentence) + "\n"))

    train_tokens = [item for sublist in swhole_tokens for item in sublist]
    whole_tokens = [item for sublist in strain_tokens for item in sublist]

    model = fasttext.train_unsupervised('wikipedia_corpus_clean.txt', model='skipgram', verbose=True)
    model.save_model("models/model_wiki_clean.bin")
    model = Word2Vec(sentences=whole_tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_wiki_clean.model")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus_clean.txt', model='skipgram', verbose=True)
    model.save_model("models/model_train_clean.bin")
    model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_train_clean.model")


if __name__ == '__main__':
    create()
