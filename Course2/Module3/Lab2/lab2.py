#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Embeddings
Created on Sat Jan 24 20:07:57 2026

@author: ek
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations

import helper_utils

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

#%% # Download the data for the GloVe 6B 100d model
helper_utils.download_glove6B()

#%% # Specify the path to the 100d GloVe file
glove_file = './glove_data/glove.6B.100d.txt'

# Load the pre-trained word vectors from the file
glove_embeddings = helper_utils.load_glove_embeddings(glove_file)

#%% 
def find_closest_words(embedding, embeddings_dict, exclude_words=[], top_n=5):
    """
    Finds the N most semantically similar words to a given vector and their scores.

    Args:
        embedding: The vector representation of the target word.
        embeddings_dict: A dictionary mapping words to their embedding vectors.
        exclude_words: A list of words to exclude from the search.
        top_n: The number of most similar words to return.

    Returns:
        A list of tuples, where each tuple contains a word and its
        cosine similarity score, sorted in descending order of similarity.
        Returns None if the vocabulary is empty after exclusions.
    """
    # Filter the vocabulary to remove any words in the exclude list.
    filtered_words = [word for word in embeddings_dict.keys() if word not in exclude_words]
    
    # Handle the edge case where the filtered vocabulary is empty.
    if not filtered_words:
        return None
        
    # Create a matrix of all word vectors for efficient computation.
    embedding_matrix = np.array([embeddings_dict[word] for word in filtered_words])
    
    # Reshape the target embedding to a 2D array for the similarity function.
    target_embedding = embedding.reshape(1, -1)
    
    # Calculate cosine similarity between the target and all other words.
    similarity_scores = cosine_similarity(target_embedding, embedding_matrix)
    
    # Get the indices of the top N words with the highest similarity scores.
    closest_word_indices = np.argsort(similarity_scores[0])[::-1][:top_n]
    
    # Create a list of (word, score) tuples for the top N closest words.
    return [(filtered_words[i], similarity_scores[0][i]) for i in closest_word_indices]

#%% # Ensure the words exist in the glove_embeddings
if all(word in glove_embeddings for word in ['king', 'man', 'woman']):
    king = glove_embeddings['king']
    man = glove_embeddings['man']
    woman = glove_embeddings['woman']
    
#%% # The resulting vector for the analogy
result_embedding = king - man + woman

#%% 
# Set top N words
top_n = 5

# Find the top N closest words, making sure to exclude the inputs
closest_words_with_scores = find_closest_words(
    result_embedding, 
    glove_embeddings, 
    exclude_words=['king', 'man', 'woman'],
    top_n=top_n
)

# Check if any words were returned
if closest_words_with_scores:
    # Unpack the top result (word and score)
    top_word, top_score = closest_words_with_scores[0]

    # Print the top result in the original format with its score
    print(f"king - man + woman ≈ {top_word} (Score: {top_score:.4f})")

    # Print the other 4 results
    if len(closest_words_with_scores) > 1:
        print(f"\n--- Other Top {top_n-1} Results ---")
        # Loop through the rest of the list of tuples
        for word, score in closest_words_with_scores[1:]:
            print(f"{word} (Score: {score:.4f})")

#%% # Define your analogy

# Your analogy will follow this format:
# word1 - word2 + word3 = ???

# Define your words
word1 = 'university'
word2 = 'student'
word3 = 'school'

# Set how many top results you want to see
top_n = 5

#%% # A list of the words used in the analogy
analogy_words = [word1, word2, word3]

# Check if all the words exist in the embeddings dictionary
if all(word in glove_embeddings for word in analogy_words):
    # Get the embedding vector for each word
    embedding1 = glove_embeddings[word1]
    embedding2 = glove_embeddings[word2]
    embedding3 = glove_embeddings[word3]

    # Perform the vector arithmetic to find the resulting embedding
    result_embedding = embedding1 - embedding2 + embedding3

    # Find the top N closest words to the result, excluding the input words
    closest_words_with_scores = find_closest_words(
        result_embedding, 
        glove_embeddings, 
        exclude_words=analogy_words,
        top_n=top_n
    )

    # Check if the function returned any similar words
    if closest_words_with_scores:
        # Get the top word and its similarity score
        top_word, top_score = closest_words_with_scores[0]

        # Print the analogy and its top result
        print(f"'{word1}' - '{word2}' + '{word3}' ≈ '{top_word}' (Score: {top_score:.4f})")

        # Check if there are other results to display
        if len(closest_words_with_scores) > 1:
            print(f"\n--- Other Top {len(closest_words_with_scores) - 1} Results ---")
            # Loop through the rest of the results and print them
            for word, score in closest_words_with_scores[1:]:
                print(f"{word} (Score: {score:.4f})")
    else:
        # Message if no similar words were found
        print("Could not find any similar words for the given analogy.")

else:
    # Find and report which words are missing from the vocabulary
    missing_words = [word for word in analogy_words if word not in glove_embeddings]
    print(f"Error: The following word(s) were not found in the vocabulary: {missing_words}")
    print("Please try different words.")
    
#%% Visualization
# Define the vocabulary of words from different categories to visualize.
words_to_visualize = ['car', 'bike', 'plane',      # Category: Vehicles
                      'cat', 'dog', 'bird',        # Category: Pets
                      'orange', 'apple', 'grape'   # Category: Fruits
]

# A dictionary grouping the same words from `words_to_visualize` by category for easy visualization
visualization_dict = {
    'Vehicle': ['car', 'bike', 'plane'],
    'Pet': ['cat', 'dog', 'bird'],
    'Fruit': ['orange', 'apple', 'grape']
}

#%% # Initialize an empty list to store the vectors.
embedding_vectors_list = []

# Loop through each word in the `words_to_visualize` list.
for word in words_to_visualize:
    # Get the embedding for the word and add it to the list.
    embedding_vectors_list.append(glove_embeddings[word])

# Convert the list of vectors into a NumPy array.
embedding_vectors = np.array(embedding_vectors_list)

#%% Use PCA to reduce the 100-dimensional embedding vectors down to just two dimensions so they can be plotted.
# Initialize the PCA model to reduce dimensions to 2
reducer = PCA(n_components=2)

# Apply PCA to the embedding vectors to get 2D coordinates
coords_2d = reducer.fit_transform(embedding_vectors)

#%% Plot the reduced 2D embeddings to see how the words are clustered.
helper_utils.plot_embeddings(coords=coords_2d, 
                             labels=words_to_visualize,
                             label_dict=visualization_dict,
                             title='GloVe Pre-Trained Embeddings'
                            )

#%% Building Your Own Embeddings From Scratch
# Define the vocabulary list. This will contain all the words from different semantic categories that your model will learn.
vocabulary = ['car', 'bike', 'plane', 
              'cat', 'dog', 'bird', 
              'orange', 'apple', 'grape']

#%% # Create word-to-index mapping
# Initialize an empty dictionary for the word-to-index mapping
word_to_idx = {}

# Loop through the vocabulary list with an index
for i, word in enumerate(vocabulary):
    # Assign each word to its corresponding index
    word_to_idx[word] = i
    
# Create index-to-word mapping
# Initialize an empty dictionary for the index-to-word mapping
idx_to_word = {}

# Loop through the items of the newly created word_to_idx dictionary
for word, i in word_to_idx.items():
    # Assign each index to its corresponding word
    idx_to_word[i] = word
    
#%% # Get the total number of unique words in your vocabulary
vocab_size = len(vocabulary)

# Define the size of the embedding vector for each word
embedding_dim = 3

#%% Generating Training Pairs
# Define the vocabulary, grouped by semantic category.
vocab_categories = {
    'Vehicles': ['car', 'bike', 'plane'],
    'Pets': ['cat', 'dog', 'bird'],
    'Fruits': ['orange', 'apple', 'grape']
} 

#%% # Initialize an empty list to hold the training pairs.
training_pairs = []

# Iterate through the lists of words in the vocab_categories dictionary.
for category_list in vocab_categories.values():
    # Generate all permutations of 2 words from the list and add them to the training_pairs.
    training_pairs.extend(list(permutations(category_list, 2)))

# Display the total number of pairs and a sample of the generated pairs.
print(f"Generated {len(training_pairs)} training pairs.")
print("Generated pairs:\n")
for pair in training_pairs:
    print(pair)

#%% Embedding Model
# Now that you have your training data, you need a model that can learn from it. You'll create a simple neural network with two key layers:
# nn.Embedding: It acts as a lookup table where each word's index (like 3 for 'cat') maps to a specific vector. During training, the model uses this vector to make a prediction and then adjusts it to better capture the word's relationships, turning the initially random vector into a meaningful one.
# nn.Linear: This is a standard fully connected layer. It will take the word vector from the embedding layer and produce a prediction for the target word.
class SimpleEmbeddingModel(nn.Module):
    """A simple neural network model for learning word embeddings."""
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the layers of the model.

        Args:
            vocab_size: The total number of unique words in the vocabulary.
            embedding_dim: The desired dimensionality of the word embeddings.
        """
        # Call the constructor of the parent class (nn.Module).
        super().__init__()
        
        # An embedding layer that maps word indices to dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # A linear layer that projects the embedding vector to the vocabulary size.
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: A tensor of input word indices.

        Returns:
            A tuple containing the output logits from the linear layer and
            the intermediate embedding vectors.
        """
        # The input 'x' is passed through the embedding layer.
        embedded = self.embedding(x)
        # The resulting embedding vector is passed through the linear layer.
        output = self.linear(embedded)
        
        return output, embedded
    
#%% Initialize the model.
embedding_model = SimpleEmbeddingModel(vocab_size, embedding_dim)

#%% Optimizer and Loss Function
# Define the Adam optimizer.
# Define Cross Entropy as the loss function.
# Initialize the Adam optimizer
optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)

# Initialize the CrossEntropyLoss function
loss_function = nn.CrossEntropyLoss()

#%% Training Embeddings with a Simple Model
# This training loop follows the same core logic you've seen before: it iterates for multiple epochs, performs a forward pass, calculates loss, and uses a backward pass to update the model.
# The key difference is how it handles the data. Instead of working with images and labels, this loop is designed for word pairs.
# Data Iteration: Rather than using a DataLoader that yields batches of (image, label), this loop iterates directly over your simple list of (word1, word2) pairs.
# Text-to-Index Conversion: The most significant difference is the data preparation step inside the loop. For each pair, the string words (e.g., 'car', 'bike') must be converted into their numerical indices using the word_to_idx mapping. This is the essential step for processing text.
# Input and Target: In this setup, the index of the first word in the pair (word1_idx) is the input to the model. The index of the second word (word2_idx) acts as the target label that the loss function uses to measure error.
def training_loop(model, training_pairs, epochs=2000):
    """
    Trains a simple word embedding model.

    Args:
        model: The PyTorch model to be trained.
        training_pairs: A list of tuples, where each tuple is an
                        (input_word, target_word) pair.
        epochs: The total number of training iterations over the dataset.

    Returns:
        A tuple containing the trained model and a list of the average
        loss for each epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize a list to store the loss value for each epoch.
    losses = []

    # Loop over the dataset for a specified number of epochs.
    for epoch in range(epochs):
        # Initialize the total loss for the current epoch.
        epoch_loss = 0

        # Loop through each input-target pair in the training data.
        for word1, word2 in training_pairs:
            # Convert the string words into their corresponding numerical indices.
            word1_idx = torch.tensor([word_to_idx[word1]])
            word2_idx = torch.tensor([word_to_idx[word2]])

            # Perform a forward pass to get the model's predictions.
            output, _ = model(word1_idx)
            # Calculate the loss between the predictions and the actual target.
            loss = loss_function(output, word2_idx)

            # Clear any previously calculated gradients before the backward pass.
            optimizer.zero_grad()
            # Compute the gradient of the loss with respect to model parameters.
            loss.backward()
            # Update the model's weights based on the computed gradients.
            optimizer.step()

            # Accumulate the loss for the current epoch.
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch and store it.
        losses.append(epoch_loss / len(training_pairs))

        # Periodically print the training progress.
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

    # Print the final loss after training is complete.
    print(f"Epoch {epochs}, Loss: {losses[-1]:.4f}")

    return model, losses

#%% Train
trained_model, losses = training_loop(embedding_model, training_pairs)

#%% Plot loss values
helper_utils.plot_loss(losses)

#%% Exploring Semantic Similarity
# Now that the model is trained, you can check if it has learned meaningful relationships. You'll do this using the same cosine similarity metric you explored earlier with the GloVe embeddings.
# The goal is to see high similarity scores for words within the same category (e.g., 'car' and 'bike') and low scores for words in different categories (e.g., 'car' and 'cat'). This will confirm that the training was successful.
# Define the cosine_similarity_words function.
# The function will take two words as input and return their similarity score based on the learned embeddings.
def cosine_similarity_words(word1, word2, word_to_idx, embeddings_matrix):
    """
    Calculates the cosine similarity between two words.

    Args:
        word1 (str): The first word to compare.
        word2 (str): The second word to compare.
        word_to_idx (dict): A mapping of words to their indices.
        embeddings_matrix (np.ndarray): The matrix containing all word vectors.
    """
    idx1 = word_to_idx[word1]
    idx2 = word_to_idx[word2]

    emb1 = embeddings_matrix[idx1]
    emb2 = embeddings_matrix[idx2]

    similarity = cosine_similarity([emb1], [emb2])[0][0]

    return similarity

#%% Pull all the final word vectors from the trained_model. The most efficient way is to grab the entire matrix of embeddings at once.
# trained_model.embedding.weight: Directly accesses the final, learned word vectors from the embedding layer's weight matrix.
# .detach().numpy(): Converts the vector matrix from a PyTorch tensor into a NumPy array, making it easier to work with.
# Set the model to evaluation mode.
trained_model.eval()

# Extract the embedding matrix.
all_embeddings = trained_model.embedding.weight.detach().numpy()

#%% Define test pairs.
similarity_tests = [
    ("car","car"),
    ("car","bike"),
    ("car","plane"),

    ("car","cat"),
    ("car","dog"),
    ("car","bird"),

    ("car","orange"),
    ("car","apple"),
    ("car","grape"),

]

#%% Finally, loop through the test pairs, calculate the similarity score for each one, and print the results.
print("Semantic Similarity (Cosine Similarity):")
print("="*40)

# Loop through each pair of words in the test list.
for word1, word2 in similarity_tests:
    # Calculate the similarity score for the current pair
    similarity = cosine_similarity_words(word1, word2, word_to_idx, all_embeddings)
    
    # Print the word pair and their calculated similarity
    print(f"{word1} <-> {word2}:\t {similarity:.4f}")
    
#%% To get a more comprehensive view than just testing a few pairs, calculate the similarity score between every possible combination of words in your vocabulary.
# Initialize an empty square matrix with zeros to hold the similarity scores.
similarity_matrix = np.zeros((vocab_size, vocab_size))

# Iterate through each row of the matrix (representing the first word).
for i in range(vocab_size):
    # Iterate through each column (representing the second word).
    for j in range(vocab_size):
        # For any word compared with itself, the similarity is a perfect 1.0.
        if i == j:
            similarity_matrix[i, j] = 1.0
        # For pairs of different words:
        else:
            # Get the string representation of each word from their indices.
            word1 = idx_to_word[i]
            word2 = idx_to_word[j]
            # Calculate the similarity and place it in the correct cell of the matrix.
            similarity_matrix[i, j] = cosine_similarity_words(word1, word2, word_to_idx, all_embeddings)
            
#%% Plot similarity matrix
helper_utils.plot_similarity_matrix(similarity_matrix, vocabulary)

#%% Visualizing Your Word Embeddings
# Reduce dimensionality
reducer = PCA(n_components=2)
coords = reducer.fit_transform(all_embeddings)

#%% Plot
helper_utils.plot_embeddings(coords=coords, 
                             labels=vocabulary,
                             label_dict=vocab_categories,
                             title='Your Trained Word Embeddings'
                            )

#%% Beyond Static Embeddings: The Importance of Context
# The sentences for comparison
sentence1 = "A bat flew out of the cave."
sentence2 = "He swung the baseball bat."

#%% # Get the specific vectors for "bat" from each sentence
bat_from_sentence1 = glove_embeddings["bat"]
bat_from_sentence2 = glove_embeddings["bat"]

#%% # --- Print vectors for the first sentence ---
print("--- Sentence 1 (first 5 values) ---")
for word in sentence1.split():
    # Clean the word to remove common punctuation
    clean_word = word.strip('.,?!').lower()
    
    # Check if the clean word exists in the GloVe vocabulary
    if clean_word in glove_embeddings:
        vector = glove_embeddings[clean_word]
        print(f"{clean_word:<12} {vector[:5]}")
    else:
        print(f"{clean_word:<12} {'(not in vocabulary)'}")

#%% # --- Print vectors for the second sentence ---
print("--- Sentence 2 (first 5 values) ---")
for word in sentence2.split():
    # Clean the word to remove common punctuation
    clean_word = word.strip('.,?!').lower()
    
    # Check if the clean word exists in the GloVe vocabulary
    if clean_word in glove_embeddings:
        vector = glove_embeddings[clean_word]
        print(f"{clean_word:<12} {vector[:5]}")
    else:
        print(f"{clean_word:<12} {'(not in vocabulary)'}")
        
#%% # Check if the two vectors for "bat" are identical
are_identical = np.array_equal(bat_from_sentence1, bat_from_sentence2)
print(f"Are the vectors for 'bat' from each sentence identical? {are_identical}")

#%% The Solution: Contextual Embeddings with BERT (Bidirectional Encoder Representations from Transformers)
helper_utils.download_bert()

#%% Load the tokenizer and the model into their respective variables.
# Define the local path where the BERT model is saved.
bert_path = './bert_model'

# Load the tokenizer and model from the specified path.
tokenizer, model_bert = helper_utils.load_bert(bert_path)

#%% # --- Process and Print Vectors for Sentence 1 ---
print("--- Sentence 1 (first 5 values) ---")
# Tokenize the sentence and get the model's output
inputs1 = tokenizer(sentence1, return_tensors='pt')
with torch.no_grad():
    outputs1 = model_bert(**inputs1)
last_hidden_state1 = outputs1.last_hidden_state[0] # Embeddings for all tokens

# Get the actual tokens from their IDs
tokens1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0])

# Loop through each token and its corresponding vector
for token, vector in zip(tokens1, last_hidden_state1):
    # Print the token and the first 5 dimensions of its contextual vector
    print(f"{token:<12} {vector.numpy()[:5]}")
    
#%% # --- Process and Print Vectors for Sentence 2 ---
print("--- Sentence 2 (first 5 values) ---")
# Tokenize the sentence and get the model's output
inputs2 = tokenizer(sentence2, return_tensors='pt')
with torch.no_grad():
    outputs2 = model_bert(**inputs2)
last_hidden_state2 = outputs2.last_hidden_state[0] # Embeddings for all tokens

# Get the actual tokens from their IDs
tokens2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0])

# Loop through each token and its corresponding vector
for token, vector in zip(tokens2, last_hidden_state2):
    # Print the token and the first 5 dimensions of its contextual vector
    print(f"{token:<12} {vector.numpy()[:5]}")
    
#%% # Extract the vector for "bat" from the first sentence (at token index 2)
bat_animal_vector = last_hidden_state1[2].numpy()
# Extract the vector for "bat" from the second sentence (at token index 5)
bat_sport_vector = last_hidden_state2[5].numpy()
# Check if the two contextual vectors for "bat" are identical
are_identical = np.array_equal(bat_animal_vector, bat_sport_vector)
print(f"Are the contextual BERT vectors for 'bat' identical? {are_identical}")

