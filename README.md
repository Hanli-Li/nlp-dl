# nlp-dl

This repository is based on Natural Language Processing with Deep Learning by Christopher Manning and Richard Socher at Stanford University. There are three independent tasks: word2vec, dependency parsing, named-entity recognition.

## word2vec

This task implements skip-gram model, which predicts context words given target word, and continuous bag of words (CBOW) model, which predicts target word from bag-of-words context. Two loss functions are explored here, one is softmax with cross-entropy loss, the other is [negative sampling loss](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). Stochasitic gradient descent is used to train word vectors on these loss functions.

The trained word vectors are applied to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, the average of all the word vectors in that sentence is used as its feature, and sentiment level of a sentence is predicted as one of five classes from very negative to very positive.

## dependency parsing

The goal of this task is to implement a dependency parser. A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads. The implementation here is a transition-based parser, which incrementally builds up a parse one step at a time. At every step it maintains a partial parse that consists of a stack of words that are currently being processed, a buffer of words yet to be processed, a list of dependencies predicted by the parser. Initially, the stack only contains ROOT, the dependencies lists is empty, and the buffer contains all words of the sentence in order. At each step, the parse applies a transition to the partial parse until its buffer is empty and only ROOT is on the stack. There are three possible transitions, SHIFT, which removes the first word from the buffer and pushes it onto the stack; LEFT-ARC that marks the second most recently added item on the stack as a dependent of the first item and removes the second item from the stack; RIGHT-ARC, which marks the first item on the stack as a dependent of the second item and removes the first item from the stack.

To decide among transitions at each state, a neural network classifier is used. And minibatch dependency parsing algorithm is implemented to make neural network run much more efficiently, as it predicts the next transition for many different partial parses simultaneously. The feature used here is based on this [neural dependency parsing paper](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf). The network consists of an embedding layer, a hidden layer with ReLU activation, and softmax output layer. The network is trained by [Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf) on cross-entropy loss function. Parameters are initialized via [Xavier initialization](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf), and network is regularized with [Dropout](https://www.cs.toronto.edu/hinton/absps/JMLRdropout.pdf)

## named-entity recognition

This task explores several neural networks for named-entity recognition. There are four special entities: Person, Organization, Location, Miscellaneous. And prediction scores are evaluated on both token level and named-entity level.

The baseline model is window-based. For each word in a sentence, its label is predicted separately using features from a window around it. Every sentence is padded for words at the beginning and the end of the sentence. A simple feedforward neural network is used here. The network contains an embedding layer, a hidden layer with ReLU activation, an softmax output layer. The network is regularized with Dropout and trained with an Adam optimizer on cross-entropy loss of the output.

The window-based model cannot use information from neighboring predictions to disambiguate labeling decisions, leading to non-contiguous entity predictions. It also neglects information from other parts of the sentence. So vanilla RNN and RNN with GRU are explored. For RNNs, sentences are padded to have the same length so that RNN can be unrolled the same number of times for different sentences. Mask vectors are used later to ignore the predictions the network makes on the padded tokens. Other details such as Dropout, Adam optimizer, Xavier initialization are the same as the baseline model.
