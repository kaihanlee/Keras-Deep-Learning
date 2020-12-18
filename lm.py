from readers.ngram_dataset import NGramDataset
from models.ngram_nlm import LanguageModel
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_history(history):
    loss = history.history['loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(x, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()


def exercises_sonnet():
    # Exercise 1: Fit a 3-gram Language Model on the Sonnet dataset. Plot the training loss histogram.
    # You will only need to implement LanguageModel.__init__() and LanguageModel.train().
    # To help you get started, we are giving you the scaffolding for calling the dataset loader, model builder, training
    # and plotting.

    # Init hyperparameters
    corpus = "sonnet"
    ngram_size = 3
    epochs = 10
    batch_size = 8

    # Create dataset of ngrams
    dataset = NGramDataset(corpus=corpus, ngram_size=ngram_size)
    print(dataset.X)
    # Create Language Model (LM)
    lm = LanguageModel(ngram_size, dataset.tokenizer)
    # Train LM
    history = lm.train(dataset.X, epochs=epochs, batch_size=batch_size)
    # Plot training curve
    # TODO: Uncomment once you have implemented the model!
    plot_history(history)


    # Exercise 2: Predict the next word and probability score given the bigram 'all the'.
    # You will need to implement LanguageModel.predict() AND call it below with 'context_vec' as the argument.
    context = ['all the']
    context_vec = dataset.vectorize(context)
    pred_index, logits = lm.predict(context_vec)
    # TODO:  Complete the Implementation


    # Exercise 3: Generate some random text (20 words), starting with '<sos> <sos>'. (Optional) you can try with
    # different initial contexts!
    # You will need to implement LanguageModel.generate()
    context = ['<sos> <sos>']
    context_vec = dataset.vectorize(context)
    text = lm.generate(context_vec)
    print(" ".join(text))


def exercises_toy():
    # Exercise 4a: Fit a 3-gram Language Model on the Toy Dataset. Plot the training loss histogram.
    # You will need to write the scaffolding below on your own.
    #TODO: Complete the implementation
    dataset = None
    lm = None

    # Exercise 4b: Predict the next word given the bigram '<sos> the'.
    # Why is the next word 'thief' and not 'crook'? Justify by looking at the dataset and the predicted probabilities.
    # You will need to write the scaffolding below on your own.
    # TODO: Complete the Implementation


    # Exercise 5: Which of the two sentences S1: 'The thief stole the suitcase.' and S2: 'The crook stole the suitcase.'
    # is more likely? Justify by looking at the dataset and computing the sentence likelihoods.
    # You will need to implement LanguageModel.sent_log_likelihood()
    sentence_1 = 'The thief stole the suitcase.'
    sentence_2 = 'The crook stole the suitcase.'
    sent_1_ngrams = dataset.get_ngrams(dataset.vectorize([dataset.preprocess(sentence_1)])[0])
    sent_2_ngrams = dataset.get_ngrams(dataset.vectorize([dataset.preprocess(sentence_2)])[0])
    log_likelihood_1 = lm.sent_log_likelihood(sent_1_ngrams)
    log_likelihood_2 = lm.sent_log_likelihood(sent_2_ngrams)
    print("P(S1)=", log_likelihood_1)
    print("P(S2)=", log_likelihood_2)

    # Exercise 6: Trained Word Embeddings. Which embeddings are closer? Compare the embeddings of 'thief' and 'crook'
    # to 'thief' and 'cop'. Justify your answer by computing the cosine similarities. Does the result follow the dataset?
    # You will need to implement the cosine similarity. Use lm.get_word_embedding() to get the word embeddings.

    # TODO: Implement cosine similarity


    # (MSc students only!) Exercise 7: Sentence Completion. Given a sentence with a gap predict the most likely word to
    # fill it in.
    #  a. Given S1: 'The [gap] was arrested by the detective.', what is the most likely predicted word?
    #  b. What is more likely to fill the gap: 'cop' or 'crook'?
    # Justify by looking at the dataset and by comparing the word embeddings pairs ('thief', 'crook') and ('thief', 'cop').
    # You will need to implement LanguageModel.fill_in() and use LanguageModel.get_word_embedding().
    sentence = 'The [gap] was arrested by the detective.'
    sentence = dataset.preprocess(sentence)
    prefix, suffix = sentence.split('[gap]')
    prefix_vec = dataset.vectorize([prefix])[0]
    suffix_vec = dataset.vectorize([suffix])[0]
    pred_word_id, logits = lm.fill_in(prefix_vec, suffix_vec, get_ngrams_fn=dataset.get_ngrams)
    pred_word = dataset.tokenizer.index_word[pred_word_id]
    print("[gap]=",pred_word)


def main():

    exercises_sonnet()

    exercises_toy()


if __name__ == "__main__":
    main()