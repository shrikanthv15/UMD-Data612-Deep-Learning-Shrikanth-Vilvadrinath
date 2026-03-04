## Assignment 2 — IMDB Sentiment Classification

In this assignment, you will use the IMDB dataset, which is available at `http://ai.stanford.edu/~amaas/data/sentiment/`. However, you don't need to separately download it, as Keras can load it for you.

The IMDB dataset contains 50,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given movie review has a positive or negative sentiment. The data was collected by Stanford researchers and was used in their 2011 paper, *"Learning Word Vectors for Sentiment Analysis"*.

A recent breakthrough in the field of natural language processing (NLP) is called **word embedding**. This is a technique where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space. Discrete words are mapped to vectors of continuous numbers. This is useful when working with natural language problems with neural networks as we require numbers as input values. Keras provides a convenient way to convert positive integer representations of words into a word embedding via an `Embedding` layer.

Convolutional neural networks (CNN) were designed to honor the spatial structure in image data while being robust to the position and orientation of learned objects in the scene. This same principle can be used on sequences, such as the one-dimensional sequence of words in a movie review. Keras supports one-dimensional convolutions and pooling by the `Conv1D` and `MaxPooling1D` classes respectively.

### Tasks

1. **Data loading and preprocessing**
  - Import all necessary libraries and load the dataset, but only keep the top n words (zero the rest), where n = 5000.  
  - Pad the dataset to a maximum review length in words.
2. **Train–test split**
  - Split the data into train and test sets.
3. **Baseline models**
  - Predict positive and negative reviews with:
    - A logistic regression model, and  
    - A feed-forward neural network model.
4. **Convolutional neural network model**
  - Create a CNN model that uses at least one `Conv1D` and one `MaxPooling1D` layer.  
  - Make your best judgement for activation functions and optimizer.  
  - For consistency and comparison with expected results, use accuracy as the evaluation metric.
5. **Training and evaluation**
  - Fit the model and evaluate it.  
  - Print the accuracy in percentage.  
  - Make your best judgement for number of epochs and batch size.
6. **Effect of stopwords**
  - Remove stopwords from the review texts and observe how this affects performance.
7. **Analysis**
  - Write your analysis of the performance of the different models.

### Performance Expectation

Reasonable performance should generate an accuracy of **over 85%**.

### Framework and Reproducibility

Although the above steps are provided with the Keras framework (v2.3.1) in mind, you may use any well-established deep learning framework, including Keras, TensorFlow, PyTorch, etc. Running code that achieves the same set of tasks and generates an output in the expected range is what determines the quality of your code.

If you did not follow the *"Setting Up Your DL Development Environment"* document and your DL framework is not Keras, please provide full details (framework name, version, etc.) so that your environment can be reproduced when your code is tested.

### Submission Instructions

Please upload:

- Your Python code (`.py` or `.ipynb`), and  
- Your report (`.pdf`)

by the due date March 6, 2026