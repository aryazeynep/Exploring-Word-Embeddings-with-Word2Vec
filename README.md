# Exploring-Word-Embeddings-with-Word2Vec

This project implements two key NLP tasks using the pre-trained `word2vec-google-news-300` model via the Gensim library:

## Tasks

### 1. Replace with Similar Words
Given a sentence and a list of word indices, this task retrieves the top 5 most similar words for each indexed word and replaces each with one randomly selected alternative.

- **Function**: `replace_with_similar(sentence: str, indices: List[int]) -> Tuple[str, Dict[str, List[Tuple[str, float]]]]`
- **Returns**: A new sentence and a dictionary of similar words with similarity scores.

### 2. Find Similar Sentences
Computes sentence embeddings by averaging word vectors and finds the most similar sentences to a query based on cosine similarity.

- **Functions**:
  - `sentence_vector(sentence: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]`
  - `most_similar_sentences(file_path: str, query: str) -> List[Tuple[str, float]]`

## Dependencies

- Python 3.x
- `gensim`
- `numpy`

## Pre-trained Model

```python
import gensim.downloader
model = gensim.downloader.load("word2vec-google-news-300")
