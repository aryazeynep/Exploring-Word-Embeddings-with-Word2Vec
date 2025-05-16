# Arya Zeynep Mete

# Objectives:
# Explore selected applications of word embedding vectors with codding their implementation. 
# Gensim library will be employed to work with Word2Vec representations. 
# The pre-trained word2vec-google-news-300 model will be utilized. 

import gensim.downloader
import random
import numpy as np
model = gensim.downloader.load("word2vec-google-news-300")


def replace_with_similar(sentence, indices):
    # sentence parameter will contain a single string
    # indices will be a list of indices and they are always be valid
    tokenized_sentence = sentence.split() # sentence will be tokenized based on whitespace
    
    words_list = [tokenized_sentence[i] for i in indices]

    words_dictionary = {word: [] for word in words_list}

    for word in words_list:
        words_dictionary[word] = model.most_similar(word, topn=5) # the top 5 most similar words along with their similarity scores
        #Word2Vec.most_similar(positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None)

    randomly_selected_words = [] # one of the five words for each token will be randomly selected 
    for word in words_list:
        randomly_selected_words.append(random.choice(words_dictionary[word])) 

    for i, word in zip(indices, randomly_selected_words):
        tokenized_sentence[i] = word[0]  # word is a tuple: (similar_word, score)
                                         # replace the original word

    final_str_sentence = " ".join(tokenized_sentence)
    return final_str_sentence, words_dictionary # return both the dictionary and the new sentence 



def sentence_vector(sentence):
    # take only a sentence string and will store the Word2Vec vectors of each word in the string in a dictionary. 
    tokenized_sentence = sentence.split() # sentence will be tokenized based on whitespace

    vector_dict = {word: [] for word in tokenized_sentence}

    summed_vector = np.zeros(300)
    for key, value in vector_dict.items():
        if key in model:
            vector = model[key]  # This is a numpy.ndarray of shape (300,)
            vector_dict[key] = vector
        else:
            vector_dict[key] = np.zeros(300) # If word not found, add 300-dim zero array
        summed_vector = summed_vector + vector_dict[key]

    length = len(vector_dict)
    sentence_vec = summed_vector / length # mean of vectors

    return vector_dict, sentence_vec


def most_similar_sentences(file_path, query):
    # generate sentence vectors for both the query and the 20 different sentences. 
    sentences = []
    vec_sent = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            sentences.append(line)
            a, b = sentence_vector(line)
            vec_sent.append(b)
    c, query_vec = sentence_vector(query)

    # Each sentence should be paired with the query, and the cosine similarity between their sentence vectors 
    final_list = []
    for i in range(len(vec_sent)):
        # Dot product
        dot_product = np.dot(vec_sent[i], query_vec)

        # Norms
        norm_vector_i = np.linalg.norm(vec_sent[i])
        norm_vector_q = np.linalg.norm(query_vec)

        # Cosine similarity 
        cosine_similarity = dot_product / (norm_vector_i * norm_vector_q)
        tuple = (sentences[i], cosine_similarity)
        final_list.append(tuple)

    final_list = sorted(final_list, key=lambda x: x[1], reverse=True) # sort the list in descending order of cosine similarity 
    return final_list

