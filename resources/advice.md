Knowledge collection
====================

This file contains shorthand explanations of useful and important discoveries, 
tricks and pitfalls you should be aware of. All of this concerns (only) the project 
itself and can either not be found in the web resources in web_resources.md 
or is not obvious / too much work to find. In the later case a reference might be given. 

## Table of contents
1. [Preprocessing](#Preprocessing)
2. [DNN architecture](#DNN-architecture)
3. [DNN training](#DNN-training)
4. [Visualisation](#Visualisation)
5. [Python](#Python)
6. [General](#General)

Preprocessing
------------------
* Scaling and normalising data ==> Equalise the "units" of the input data.
Don't forget to take the original data range and it's meaning into account!
* Feature trade-off between usefulness and (too high) dimensions. 
* Reduce the combinatorial explosion of data variations. The same kind of property 
/ entity shouldn't look different, just because it is at another location in the data.
* t-SNE:
	- You cannot see relative sizes of clusters in a t-SNE plot.
	- Distances between well-separated clusters in a t-SNE plot may mean nothing.
	- t-SNE tends to expand denser regions of data
	- Depending on the parameters the plot might be completely misleading

DNN architecture
------------------
* LSTM forget bias ==> initialise with 1
* Better no activation function at the end. Be aware of them in general.
* Try to experiment with activation functions. SeLU, ELU, ReLU, self-normalising 
tanh, ... [ML subreddit discussion](https://www.reddit.com/r/MachineLearning/comments/6g5tg1/r_selfnormalizing_neural_networks_improved_elu/)
* Appropriate weight initialisation
* Start out with few layers and revise this NN until it works relatively fine 
and only then start adding layers

DNN training
------------------
* Sanity-check of the (intermediate) results, if possible. 
Try to understand why it isn't working; what the erroneous result means. 
E.g. Gradient explosion, gradient vanishing. Coarser: underfitting, overfitting. 
Sanity check suggestions: Only one item (of each class), see if it overfits to them; 
easy and small text classification dataset; Can you overfit to the whole training set?; 
* Play with the learning rate and gradient clipping. Also consider the influence 
of the Batch Norm regularisation on the learning rate.

Visualisation
------------------


Python
------------------


General
------------------
* Don't believe everything you have read somewhere without mathematical proof, 
correct logical reasoning or genuine empirical evidence. Countercheck if 
necessary with other sources. If you don't know better but it seems useful and 
correct, write it down here and correct it if necessary.
