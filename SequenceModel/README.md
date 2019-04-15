# Using numpy to build sequence models using RNN, LSTM and GRN

## Case study: Generate text data
There are word-based and character-based language models. In this example, we used the character-based model. This is because for word-based model, we require a larger dataset of text data to train. As we are using numpy only, we also want to avoid using a large vector for the volcaburary for simplicity purpose. 

Compared to word-based model, character-based model doesn't have the problem for unknown words. However, character-based model, is not good at catching long range dependencies. This is not difficult to image, for two word, there are just two timestep apart in a word-based model, but several timesteps (characters) apart in the character-based model.

For the text generator, the purpose of the deep learning model is to learn the (at least) two weight matrixes to match the probability of the next character (word) from the previous character (word).

Once this probability is learn, the text generator only need to do a random sampling, for example, using np.random.choice to sample the character (word) using their corresponding probability. This can be understand by thinking each character (word) is an apple in the bucket. Each apple has a different probability of being pick up from the bucket. We can't see through the bucket, but we can pick up one apple at a time from the bucket. We laid down the apple in a row, which is then our produced sentence. 

## What is an RNN?
One RNN cell is not one neuron with feedback loop linking itself. One RNN cell is a neuron network layer with feedback loop from the previous cell states with the additional weight matrixs. 
<img src = RNN.png> 
<img src = RNN_cell.png> 

## What is an LSTM?
Similarly, a LSTM cell is not one neuron but a whole neural network layers with multiple neurons. Each LSTM cell have several gates:
<img src = LSTM_cell.png> 

Forget gate: can think about it as a bit operation (it is a sigmoid activation) to decide which bit to forget

Update gate: if one bit doesn't want to forget, then it will be update, also a bit operation (0 or 1). As now there is information needs to be update, we need to build up the new information, with a tanh (-1 to 1).

Output gate: decide if a bit need to be rewrite (bit operation) and what information (tanh) to write
