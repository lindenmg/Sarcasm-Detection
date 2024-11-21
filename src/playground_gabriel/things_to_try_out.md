* Semantic clustering with e.g. gensim models.lsimodel (truncated SVD) for groups of words.
  Order in Sentence as feature dimension. nltk.collocations http://www.nltk.org/howto/collocations.html
* Also, maybe highlight log_probability, number, out of vocabulary words,
* out-of-vocabulary or not, optimise word coverage with spacy nlp.vocab.prune_vectors()
* Look up/try out, if Batch Renorm sensible
* Some kind of outlier detection might be useful
* 3D convolution with lemma-dimension
* In case of too much time also try to get attention in case of no RNNs to work
* See how many one, or two word replies there are and maybe kick them out

Attention:

* "So now I have stuff like a variant which does something like a deep kNN (distance-based attention) to make something
  somewhat robust to nonstationarity in timeseries prediction,"
* Try PCA for Conv to scalar attention + gate. To reduce word-vectors to 1-dim.
  Only if this AttentiveConv is reasonable good anyway.

comparison with paper:
"Before constructing this subcorpus we first remove from
consideration all comments that are not complete sentences
and not between 2 and 50 tokens long, allowing for cleaner
comments in the evaluation" ==> Not the case in our corpus!
But ours is balanced.

Check at some point in the future how small the weights of a NN can get.
Because of numerical instability regarding significant values.
(Mantissa <--> Exponent)

Some examples are not English
Quite a few false negatives and some should be false positives

