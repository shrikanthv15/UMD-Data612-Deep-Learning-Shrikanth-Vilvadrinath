# Assignment 3: LSTM Sequence Prediction

## Background

- LSTM is widely used because its architecture overcomes the vanishing and exploding gradient problem that affects recurrent neural networks.
- This allows very large and very deep networks to be created.

## Objective

- Develop an LSTM recurrent neural network model for a simple sequence prediction task: learning the alphabet.
- Given a letter of the alphabet, predict the next letter.
- This problem can be generalized to other sequence prediction tasks such as:
  - Time series prediction
  - Sequence classification

## Problem Setup

- Build a simple LSTM model that predicts the next character in the alphabet using the context of one character.
- Frame the task as a random collection of one-letter input to one-letter output pairs.

## Implementation Steps

1. Import all necessary libraries and define the dataset, such as:
   - `alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"`
2. Create a mapping of characters to integers (0–25) and the reverse mapping.
3. Prepare the dataset of input-output pairs encoded as integers.
4. Reshape `X` to `[samples, time steps, features]`.
5. Normalize the data.
6. One-hot encode the output variable (convert it to categorical format).
7. Create and fit the model.
8. Summarize model performance by printing accuracy.
9. Demonstrate predictions by feeding input characters and printing predicted output characters.
10. Run experiments with network depth:
    - Start with two LSTM layers using the same parameters.
    - Vary the number of layers and compare performance.
    - Include plots/graphs/tables to support your claim.
11. Run hyperparameter experiments:
    - Vary learning rate, hidden size, and number of layers.
    - Analyze how each parameter affects overall accuracy.
    - Include plots/graphs/tables for each hyperparameter and explain how you selected the optimal values.

## Expected Performance

- Reasonable performance should achieve an accuracy over 80%.

## Submission

- Upload your Python code (`.py` or `.ipynb`) and report (`.pdf`) by the due date.
