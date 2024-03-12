# NLP_Study
NLP_Study
Week 1 - Neural Network Basic Foundation.
1. Classification/regression model
2. Use pytorch / pure python(extra credit)
3. Use two layers including activation functions at least (linear layer, relu etc…)
4. Have a good understanding of backward and forward propagation process during the training and have a short summary.
5. Have a mathematical summary (model structure) of your own model
6. Have a base model as a benchmark ex) logistic/regression model
7. Add regularization techniques in your loss function on your base and nn model such as L1 and L2 and explain why you used it.
8. Have your github page shown these results in a notebook.(math expression in github as well)

Week 1 sources:
https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
https://towardsdatascience.com/how-to-define-a-neural-network-as-a-mathematical-function-f7b820cde3f
https://www.analyticsvidhya.com/blog/2021/04/estimation-of-neurons-and-forward-propagation-in-neural-net/


Week 2 - NLP Basic Foundation
1. NLP basic components - lemmatization, token, stopwords, ngram, tfidf, tokenizer, and etc
- Summarize definition
2. Sentiment analysis - dictionary look for an open source definition and apply to the given data in your own perspective.
3. Word cloud - What are the most common words? What would be removed and what would be the result after removing?
4. Bigram, trigram Analysis - Same as 2 and 3
5. Use Tf-idf or bag of words and have your corpus.
- Summarize definition
6. Using the result from #5, predict category or section of each article. Share the result and metrics. (You can use any classification model.)
Embedding Layer: BERT begins with an embedding layer, which converts input tokens into dense vectors of fixed size. These embeddings capture semantic meaning and contextual information about the words in the input sequence.

Week 2 sources:
https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/
https://huggingface.co/docs/transformers/main_classes/tokenizer
https://realpython.com/python-nltk-sentiment-analysis/

Week 3 - NLP Basic Foundation
Deep learning NLP
Understanding the deep learning models in NLP
1. gan
2. transformer, sentence transformer
3. lstm
4. rnn
5. bidirectional lstm
6. t5
7. Tokenizers - wordpiece, unigram, byte pair etc
Through 1-7, Understand the details of each model such as network, unique technique and summarize them.

Given data set and techniques, perform ner. Then, go back to your preferred model(pretrained) from the top list and fine-tune your model and do prediction. (Add linear layer+dropout or activation function to your deep learning model and freeze the embedding layer)

Linear (Dense) Layers: BERT includes linear layers, also known as fully connected layers, which are used for dimensionality reduction or transformation of the hidden representations produced by the Transformer layers. These linear layers may be added on top of the Transformer outputs to project the representations into a different space or to reduce their dimensionality.

Activation Functions: Activation functions introduce non-linearity into the network, allowing it to learn complex relationships between inputs and outputs. Common activation functions include ReLU (Rectified Linear Unit), which introduces non-linearity by outputting the input directly if it is positive and zero otherwise, and Gelu (Gaussian Error Linear Unit), which introduces a smooth non-linearity. These activation functions are typically applied after linear transformations within the network.

Normalization Layers: BERT employs layer normalization, which normalizes the activations of each layer across the feature dimension. This helps stabilize training and improve the convergence of the network.

Dropout: Dropout is a regularization technique used to prevent overfitting by randomly dropping a proportion of the units (inputs) in the network during training. BERT uses dropout to regularize the hidden representations produced by the Transformer layers or other linear layers. 

Week3 sources:
Tokenizer - https://huggingface.co/learn/nlp-course/en/chapter6/1?fw=pt
LSTM, RNN - https://www.theaidream.com/post/introduction-to-rnn-and-lstm
biLSTM - https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/
transformer - https://serokell.io/blog/transformers-in-ml
https://www.columbia.edu/~jsl2239/transformers.html
gan - https://www.geeksforgeeks.org/generative-adversarial-network-gan/
t5 - https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part
