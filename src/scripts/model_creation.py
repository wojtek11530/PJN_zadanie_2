import spacy
from gensim.models import Word2Vec
import fasttext


def create():
    with open("wikipedia_corpus.txt") as file:
        lines = file.readlines()
        whole = [line.rstrip() for line in lines]
    
    with open("data_tripadvisor_corpus.txt") as file:
        lines = file.readlines()
        train = [line.rstrip() for line in lines]
    
    nlp = spacy.load("pl_core_news_md", disable=["parser", "ner"])

    #model = fasttext.train_unsupervised('wikipedia_corpus.txt', verbose=True, model='skipgram')
    #model.save_model("models/model_wiki_base_skip.bin")
    #model = fasttext.train_unsupervised('wikipedia_corpus.txt', verbose=True, model='cbow')
    #model.save_model("models/model_wiki_base_bow.bin")
    wiki_worlds = [sentance.split(" ") for sentance in whole[:int(len(whole)/8)]]
    model = Word2Vec(sentences=wiki_worlds, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_wiki_base.model")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus.txt', model='skipgram', verbose=True)
    model.save_model("models/model_train_base_skip.bin")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus.txt', model='cbow', verbose=True)
    model.save_model("models/model_train_base_cbow.bin")
    train_worlds = [sentance.split(" ") for sentance in train]
    model = Word2Vec(sentences=train_worlds, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_train_base.model")

    strain_tokens = []
    for sentence in train:
        worlds = nlp(sentence)
        strain_tokens.append(
                [
                    token.lemma_.lower()
                    for token in worlds
                    if not (
                        token.is_stop
                        or token.is_punct
                        or token.like_email
                        or token.like_url
                        or token.like_num
                        or token.is_digit
                        or token.pos_ not in ["NOUN", "ADJ", "VERB", "ADV"]
                    )
                ]
            )  

    swhole_tokens = []
    for sentence in whole[:int(len(whole)/8)]:
        worlds = nlp(sentence)
        swhole_tokens.append(
                [
                    token.lemma_.lower()
                    for token in worlds
                    if not (
                        token.is_stop
                        or token.is_punct
                        or token.like_email
                        or token.like_url
                        or token.like_num
                        or token.is_digit
                        or token.pos_ not in ["NOUN", "ADJ", "VERB", "ADV"]
                    )
                ]
            )

    with open('wikipedia_corpus_clean.txt', 'a') as the_file:
        for sentence in swhole_tokens:
            the_file.write((" ".join(sentence) + "\n"))

    with open('data_tripadvisor_corpus_clean.txt', 'a') as the_file:
        for sentence in strain_tokens:
            the_file.write((" ".join(sentence) + "\n"))

    whole_tokens = swhole_tokens
    train_tokens = strain_tokens
    model = fasttext.train_unsupervised('wikipedia_corpus_clean.txt', model='skipgram', verbose=True)
    model.save_model("models/model_wiki_clean_skip.bin")
    model = fasttext.train_unsupervised('wikipedia_corpus_clean.txt', model='cbow', verbose=True)
    model.save_model("models/model_wiki_clean_bow.bin")
    model = Word2Vec(sentences=whole_tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_wiki_clean.model")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus_clean.txt', model='skipgram', verbose=True)
    model.save_model("models/model_train_clean_skip.bin")
    model = fasttext.train_unsupervised('data_tripadvisor_corpus_clean.txt', model='cbow', verbose=True)
    model.save_model("models/model_train_clean_bow.bin")
    model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec_train_clean.model")


if __name__=='__main__':
    create()
