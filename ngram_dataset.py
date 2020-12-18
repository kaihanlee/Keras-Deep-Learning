from keras.preprocessing.text import Tokenizer
import numpy as np

# We will use Shakespeare Sonnet 2
sample_corpus_sonnet = ["""When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold."""]

sample_corpus_toy_sentences = [
    "The thief stole.",
    "The thief stole the suitcase.",
    "The crook stole the suitcase.",
    "The cop took a bribe.",
    "The thief was arrested by the detective."
]


class NGramDataset:

    def __init__(self, corpus='sonnet', ngram_size=3):
        self.ngram_size = ngram_size
        self.corpus = corpus
        self.tokenizer = Tokenizer(oov_token='<UNK>')
        if corpus == 'sonnet':
            self.X = self.load(sample_corpus_sonnet)
        elif corpus == 'toy':
            self.X = self.load(sample_corpus_toy_sentences)
        else:
            raise Exception('The value for corpus should be either "sonnet" or "toy"')

    '''
        Add special 'Start Of Sentence' (<sos>) and 'End of Sentence' (<eos>) symbols in the beginning and end of sentences.
        This helps learn to predict words at the beginning and end of sentences.
    '''
    def preprocess(self, sentence):
        prefix = " ".join(["<sos>" for i in range(1, self.ngram_size)])
        suffix = " ".join(["<eos>" for i in range(1, self.ngram_size)])
        return "{} {} {}".format(prefix, sentence, suffix)

    '''
    Convert sentence(s) of words into sequences of indexes. 
    '''
    def vectorize(self, sentences):
        return self.tokenizer.texts_to_sequences(sentences)

    def get_ngrams(self, sequence):
        ngrams = []
        for i in range(0, len(sequence) - self.ngram_size + 1):
            ngrams.append([sequence[k] for k in range(i, i + self.ngram_size)])
        return ngrams

    def load(self, corpus):
        corpus_with_sos_eos = []
        for sentence in corpus:
            corpus_with_sos_eos.append(self.preprocess(sentence))

        # Tokenize and convert to indexes
        self.tokenizer.fit_on_texts(corpus_with_sos_eos)
        sequences = self.vectorize(corpus_with_sos_eos)

        # Create a list of all ngrams found in all the sentences
        ngram_list = []
        for sequence in sequences:
            ngram_list += self.get_ngrams(sequence)
        # return the tokenizer (useful for converting back from indexes to words, etc) and
        # the list of ngrams as an np.array with shape (num_of_ngrams, ngram_size)
        return np.array(ngram_list)


