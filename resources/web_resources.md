Knowledge collection
====================

This file contains hyperlinks to useful papers, blogs and other web resources.
It is split up into different sections which are self explanatory.
For each listed web resource the following is listed:
* The title of the paper, blog, ...
* Optionally: 1-3 sentences which describe the content with focus on the usefulness for this project
* Keywords which describe the content, including synonyms and different spellings of them. 
That's because somebody else would probably attribute different keywords to it. Or you self - tomorrow. 
Words in the title or the optional summary don't have to be added to the keywords
* The hyperlink to the resource (paper, blog, discussion, website)
* Optionally: Hyperlinks to websites were the resource is discussed, like the Machine Learning Subreddit.
If the main resource is a discussion, this point can contain relevant papers, blogs, ...

Please follow this rules if you add another resource. The sections are only a soft constraint.
If you are searching for a certain resource, use the search function and search for keywords.
If you couldn't find the resource here and had to look for yourself and you found something useful,
please add it here.
You can also add e.g. a blog in general without linking a certain entry. Try to avoid misspellings!

In general: Other > Data Science > Research Questions > Machine Learning > Deep Learning

New feature, information added over time: 
Approximate word count (1/2 thousand steps) and month/year publish date.
**Maybe in the future:** Rating for every visited source. Discussion about that necessary!

#### Example:
1. **Title**
    - Summary
    - Keyword I, Keyword II
    - [hyperlink]()
    - [relevant](); [relevant II]()
    
#### Keyword synonyms:
Some terms which are/could be meaning/stand for the same in this document
* Drop-out, Dropout, Drop out
* RNN, Recurrent Neural Network, Recurrent, LSTM, Long Short Term Memory, GRU
* CNN, Convolution, Convolutional Neural Network
* Deep Learning, DNN, Neural Network, Deep Neural Network
* Activation function, activation, Sigmoid, non-linear, non linear, linear, ReLU, SeLU, tanh,
leaky ReLU, nonlinearity, non-linearity, non linearity, logistic function, linear function
* Regularisation, regularization, batch-norm, Batch Norm, Batchnorm, Batch-Renorm, Batch Renorm, Batch Renormalization,
Batchrenorm, L1, L2, sparse weights, normalization, normalisation, Batch-Renormalization, Batch-Normalization
* Classifiers, classification, classify, recognition, detection
* word vectors, word-vectors, wordvectors, embeddings
* state-of-the-art, SOTA, state of the art
* tipp, tip, trick, advice, recommendation
* NLP, Natural Language Processing, Computational Linguistics
* recap, developments, inventions, progress, advancements, achievement, advance

## Table of contents
1. [Data Science introduction](#Data-Science-introduction)
2. [Data Science advanced](#Data-Science-advanced)
3. [Deep Learning blog entries](#Deep-Learning-blog-entries)
4. [Other blog entries](#Other-blog-entries)
5. [Deep Learning papers](#Deep-Learning-papers)
6. [Other papers](#Other-papers)
7. [Data Science discussions and questions](#Data-Science-discussions-and-questions)
8. [Machine Learning discussions and questions](#Machine-Learning-discussions-and-questions)
9. [Research Question Pascal](#Research-Question-Pascal)
10. [Research Question Gabriel](#Research-Question-Gabriel)
11. [Frameworks, software and data](#Frameworks-software-and-data)
12. [Other resources](#Other-resources)

Data Science introduction
---------------------------------------
1. **A Few Useful Things to Know about Machine Learning**
    - From the content: learning = representation + evaluation + optimization, correlation does not imply causation,
    theoretical guarantees are not what they seem, overfitting has many faces, ...
    - tutorial, introduction, overview, tips, tipps, explanation
    - [PDF at Washingtion University](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)  
2. **The Dangers of Overfitting or How to Drop 50 spots in 1 minute**
    - Especialy overfitting to the test set. Nothing about regularisation.
    - Regularization, regularisation, Machine Learning, ML
    - [Original blog post, working graphics](http://gregpark.io/blog/Kaggle-Psychopathy-Postmortem)
    - [Kaggle tutorial blog](http://blog.kaggle.com/2012/07/06/the-dangers-of-overfitting-psychopathy-post-mortem)
3. **Overfitting, underfitting, bias-variance trade-off**
    - Misleading modelling: overfitting, cross-validation, and the bias-variance trade-off, ...
    - regularization, regularisation, tutorial, introduction, explanation, crossvalidation
    - [Backup at archive.org](https://web.archive.org/web/20160329072744/http://blog.cambridgecoding.com/2016/03/24/misleading-modelling-overfitting-cross-validation-and-the-bias-variance-trade-off); 
    [Blog post II](https://memosisland.blogspot.de/2017/08/understanding-overfitting-inaccurate.html); 
    [ML subreddit about II](https://www.reddit.com/r/MachineLearning/comments/6u84yk/d_understanding_overfitting_an_inaccurate_meme_in); 
    [ML subreddit dicussion I](https://www.reddit.com/r/MachineLearning/comments/6zdsn9/what_exactly_is_overfitting_and_why_do_we_prefer);  
    [Underfitting vs. overfitting at scikit-learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html); 
    [Underfitting vs. overfitting at Quora](https://www.quora.com/Whats-the-difference-between-overfitting-and-underfitting)
4. **Kaggle Tutorial Blog - Interviews**
    - Tips about practical Machine Learning in terms of text classification and in general.
    Some of them are only applicable for Kaggle competitions others are generally useful. 
    Relevant weblink for this project more useful.
    - Natural Language Processing, tipps, debugging, introduction, lessons learned
    - [Competition winner interview](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors); 
    [Competition-centric interview](http://blog.kaggle.com/2015/05/07/profiling-top-kagglers-kazanovacurrently-2-in-the-world)
    - [Approaching (almost) any ML Problem](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur)
5. **Deep Learning Project Workflow**
    - What to do in which order. And tips in case of bad model performance.
    - Keywords
    - [Tips at Github](https://github.com/thomasj02/DeepLearningProjectWorkflow#if-your-training-and-test-data-are-from-the-same-distribution); 
    [Workflow at blog](https://medium.com/@erogol/designing-a-deep-learning-project-9b3698aef127)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/5xlj6v/deep_learning_project_workflow_notes_from_ngs)
6. **Practical Advice for Building Deep Neural Networks**
    - What to do in certain situations during training
    - tips, tipps, DNN, Deep Learning, Deep Neural Networks, vanishing gradient
    - [BYU company blog](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks)
7. **My Neural Network isn't working! What should I do?**
    - 08/2017 Worth it, to read it beforehand. This way you know how to prevent the problems for which there are
    solutions presented in the first place.
    - Normalization, normalisation, regularisation, regularization, preprocessing, batch, learning rate, 
    gradient vanishing, initialization, initialisation, number of parameters, number of layers, DNN, 
    Deep Learning, Deep Neural Network, tips, tipps, tricks
    - [theorangeduck blog, 4k](http://theorangeduck.com/page/neural-network-not-working)
    - [Explanation at stackoverflow](https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class/41493375#41493375)
8. **Understanding Convolutional Neural Networks for NLP**
    - 11/2015. CNN in context of NLP, linked research papers
    - DNN, Deep Neural Networks, Deep Learning, CNN, Convolution, Natural Language Processing
    - [wildml blog](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp)
9. **Convolution  explains**
    - Amazing GIF images of convolutional filters along with different padding and strides! There is also a paper.
    - CNN, DNN, Deep Neural Network, Deep Learning, tutorial, visualization, visualisation, 
    Convolutional Neural Networks, tutorial, explanation 
    - [Document at Github](https://github.com/vdumoulin/conv_arithmetic)
    - [Paper which explains arithmetic behind CNN](https://arxiv.org/abs/1603.07285)

Data Science advanced
---------------------------------------
1. **Blogs about Machine Learning**
    - Blogs of people who know quite a lot and write in general good blog posts.
    - Deep Learning, DNN, Knowledge, general, Deep Neural Network, articles, tips, tipps, advice, 
    reading comprehension, advances, recommendation, tutorials, explanations, introduction, overview, ...
    - [inFERENCe](http://www.inference.vc); [distill.pub](https://distill.pub); 
    [Off the convex path](http://www.offconvex.org); [Sebastian Ruder](http://ruder.io); 
    [Andrej Kaparthy's](https://karpathy.github.io); [The Spectator - Shakir](http://blog.shakirm.com)
2. **Modern Theory of Deep Learning: Why Does It Work so Well**
    - Summary
    - Deep Neural Network, Tutorial, Function, DNN, Analysis
    - [Medium blog](https://medium.com/mlreview/modern-theory-of-deep-learning-why-does-it-works-so-well-9ee1f7fb2808)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7l85jn/r_modern_theory_of_deep_learning_why_does_it_work); 
    [Comparable video?](http://www.cgpgrey.com/blog/how-do-machines-learn)
3. **Word embeddings in 2017: Trends and future directions**
    - Word vectors, word-vectors, news, developments, Natural Language Processing, NLP, progess, development
    - [ruder blog](http://ruder.io/word-embeddings-2017)
4. **A curated list of awesome Deep Learning for Natural Language Processing resources**
    - DNN, NLP, knowledge, collection, Deep Neural Networks
    - [Github](https://github.com/brianspiering/awesome-dl4nlp)
5. **Poincaré Embeddings for Learning Hierarchical Representations**
    - Apparently outperforms normal Euclidean word vectors! Introduces a new approach for learning hierarchical 
    representations of symbolic data by embedding them into hyperbolic space. Or more precisely into an 
    n-dimensional Poincaré ball. Due to the underlying hyperbolic geometry, this allows us to learn parsimonious 
    representations of symbolic data by simultaneously capturing hierarchy and similarity
    - word-vectors, word-embeddings, wordvectors, wordembeddings, preprocessing, pre-processing, context, meaning, NLP, 
    Natural Language Processing
    - [Github implementation](https://github.com/TatsuyaShirakawa/poincare-embedding); 
    [ML subreddit question](https://www.reddit.com/r/MachineLearning/comments/6w3oq7/d_any_advice_for_a_non_computationallinguist); 
    [ML subreddit posting](https://www.reddit.com/r/MachineLearning/comments/75cq4q/r_implementation_of_poincar%C3%A9_embeddings_for); 
    [Medium blog](https://towardsdatascience.com/facebook-research-just-published-an-awesome-paper-on-learning-hierarchical-representations-34e3d829ede7)
    - [arvix paper](https://arxiv.org/abs/1705.08039)
6. **[P] Using Evolution to find good DNN hyperparams.**
    - optimization, optimisation, hyperparameter search, Deep Neural Network, Deep Learning, evolutionary algorithms, 
    DNN, neuro evolution
    - [Blog at Medium](https://medium.com/@stathis/design-by-evolution-393e41863f98)
    - [PyTorch example at Github](https://github.com/offbit/evo-design); 
    [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/6jrmus/p_using_evolution_to_find_good_dnn_hyperparams)
7. **[P] Cheat Sheets for deep learning and machine learning**
    - Some pictures/PDFs with the most important commands 
    - frameworks, Numpy, Pandas, Keras, Tensorflow, scikit-learn, Matplotlib, 
    Neural Networks, DNN, Deep Neural Networks
    - [Markdown at Github](https://github.com/kailashahirwar/cheatsheets-ai); 
    [ML subreddit with PDF link](https://www.reddit.com/r/MachineLearning/comments/6go2n9/p_cheat_sheets_for_deep_learning_and_machine/?st=j47jjaj8&sh=7bc3de79); 
    - [Blog](https://startupsventurecapital.com/essential-cheat-sheets-for-machine-learning-and-deep-learning-researchers-efb6a8ebd2e5)
8. **Deep Learning for Natural Language Processing (without Magic)**
    - Indeed short Standford university course with (flash) videos
    - DNN, NLP, Deep Neural Networks, introduction, tutorial
    - [Course at Stanford](https://nlp.stanford.edu/courses/NAACL2013)
9. **Teacher forcing**
    - Teacher forcing is a method for quickly and efficiently training recurrent neural network 
    models that use the output from a prior time step as input.
    - Training, optimization, optimisation, RNN, Deep Learning, DNN, Deep Neural Networks
    - [Blog explanation](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks); 
    [explanation at Quora](https://www.quora.com/What-is-the-teacher-forcing-in-RNN)
    - [Regarding PyTorch](https://stackoverflow.com/questions/47077831/teacher-forcing-with-pytorch-rnn); 
    [Professor forcing paper at arvix](https://arxiv.org/abs/1610.09038)    
10. **Word Embeddings and their Challenges**
    - Summary
    - word-embeddings, word vectors, word-vectors, wordvectors, preprocessing, pre-processing, NLP, 
    Natural Language Processing, GloVe, Word2vec, CBOW, post-processing, postprocessing
    - [Aylien blog I](http://blog.aylien.com/word-embeddings-and-their-challenges); 
    [Aylien blog II](http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove)
11. **The Quartz guide to bad data**
    - A comprehensive guide to data cleaning. Only a few points relevant for this project
    - data-cleaning, data engineering, data-engineering, tutorial, explanation, tips, tipps, preprocessing, 
    preprocessing, bad data, missing values, format issues, corrupted data
    - [Blog-like post at github](https://github.com/Quartz/bad-data-guide#text-is-garbled)
    - [Data-clearning subreddit all time top](https://www.reddit.com/r/datacleaning/top/?sort=top&t=all); 
    [Common Data Pitfalls for Recurring ML Systems](http://www.willmcginnis.com/2015/12/20/common-data-pitfalls-for-recurring-machine-learning-systems)  
12. **A curated list of resources dedicated to NLP**
    - Natural Language Processing, tutorials, frameworks, introductions, explanations, collections, blogs, 
    reading lists, reading-lists, blogs, frameworks
    - [Blog-like repo at Github](https://github.com/keon/awesome-nlp#named-entity-recognition)
    - [Similar repo at Github I](https://github.com/brianspiering/awesome-dl4nlp); 
    [Similar repo at Github II](https://github.com/gutfeeling/beginner_nlp); 
    [Similar repo at Github III](https://github.com/xiamx/awesome-sentiment-analysis); 
13. **deeplearning.net**
    - Meta site with a lot of links to software, blogs, papers, datasets, reading lists, ...
    - Deep Learning, DNN, Machine Learning, Deep Neural Network, tutorials, introductions, data-sets, data sets, 
    blogs, reading lists, reading-lists, frameworks, explanations
    - [deeplearning.net](http://deeplearning.net)
                      
Deep Learning blog entries
---------------------------------------
1. **Deep Learning Achievements Over the Past Year**
    - Summary
    - Deep Neural Network, DNN, Overview, Summary, Recap, Developments, Inventions, Progress, sota, state-of-the-art, 
    state of the art
    - [statsbot blog entry](https://blog.statsbot.co/deep-learning-achievements-4c563e034257)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7la1ez/r_great_deep_learning_achievements_over_the_past)
2. **Deep Learning for NLP, advancements and trends in 2017**
    - Natural Language Processing, NLP, tips, state of the art, state-of-the-art, sota, progress, developments
    - [tryolabs blog](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017)
3. **Alchemy, Rigour and Engineering**
    - The *Black Magic* behind Deep Learning, that you have quite often no real idea, why it works so well
    - Blackbox, black-box, black box, DNN, Deep Neural Networks
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7i6r7c/d_alchemy_rigour_and_engineering)
    - [inference blog](http://www.inference.vc/my-thoughts-on-alchemy)
4. **\[R\] NIPS 2017 Notes - David Abel (Brown University)**
    - Machine Learning conference - ideas, papers, methods
    - Machine Learning conference, ideas, papers, methods, Deep Learning, DNN, Deep Neural Network, progress, 
    research, development, advance, advancement, sota, state-of-the-art, state of the art
    - [Brown University PDF](https://cs.brown.edu/%7Edabel/blog/posts/misc/nips_2017.pdf)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7j9krt/r_nips_2017_notes_david_abel_brown_university)
5. **Snark Bite: Like an AI Could Ever Spot Sarcasm**
    - "Could a computer learn to detect this nuanced form of expression? 
    Pushpak Bhattacharyya says they can — and he’s got the algorithms to prove it"
    - sarcasm detection, sracasm classification, text classification
    - [Nvidia blog](https://blogs.nvidia.com/blog/2018/01/31/ai-detect-sarcasm/)
    - [Paper about word embeddings](https://www.cse.iitb.ac.in/~pb/papers/emnlp16-sarcasm.pdf)
    
Other blog entries
---------------------------------------
1. **Title**
    - Summary
    - Keywords
    - [hyperlink]()
    - [relevant]()
    
Deep Learning papers
---------------------------------------
1. **Quasi-Recurrent Neural Network (QRNN)**
    - A little bit worse, but (far) faster RNNs then LSTMs
    - improvement, improved, advancement, sota, state-of-the-art, state of the art, DNN, Deep Learning, 
    Deep Neural Network,
    - [Saleforce arvix paper](https://arxiv.org/abs/1611.01576)
    - [PyTorch implementation at Github](https://github.com/salesforce/pytorch-qrnn); 
    [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/5dep6x/16x_faster_rnn_quasirecurrent_neural_networks)
2. **fastText**
    - A very simple, fast and very good text classifier
    - Textclassification, text-classification, text classification, sota, state-of-the-art, state of the art, n-grams, 
    shallow, not deep
    - [Better fastText at arvix](https://arxiv.org/abs/1702.05531); 
    [Original fastText at arvix](https://arxiv.org/abs/1607.01759)
    - [Official original ímplementation at Github](https://github.com/facebookresearch/fastText); 
    [Keras implementation](https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py)
3. **A Simple but Tough-to-Beat Baseline for Sentence Embeddings**
    - Natural language processing, Unsupervised Learning, NLP, sentence-vectors, sentence vectors, sentence-embeddings, 
    preprocessing, pre-processing
    - [Paper at Openreview](https://openreview.net/forum?id=SyK00v5xx)
    - [Implementation at Github](https://github.com/PrincetonML/SIF)    
4. **Convolutional Neural Networks for Text Categorization: Shallow Word-level vs. Deep Character-level**
    - text classification, classifier, text-classification, DNN, CNN, Deep Learning, Deep Neural Network
    - [Paper at arvix](https://arxiv.org/abs/1609.00718)   
5. **Very Deep Convolutional Networks for Text Classification**
    - classifier, text-classification, DNN, CNN, Deep Learning, Deep Neural Network
    - [Paper at arvix](https://arxiv.org/abs/1606.01781)
6. **A Decomposable Attention Model for Natural Language Inference**
    - Describes a good way to compose two sentences into one vector. There are apparently follow-up posts 
    for the as relevant marked blog post.
    - NLP, sentence-vector, sentence vector
    - [Paper at arvix](https://arxiv.org/abs/1606.01933)
    - [spaCy v1 & Keras implementation at Github](https://github.com/explosion/spaCy/tree/master/examples/keras_parikh_entailment); 
    [Relevant blog post](https://explosion.ai/blog/deep-learning-formula-nlp); 
    [discussion at ycombinator](https://news.ycombinator.com/item?id=12929741)
7. **\[D\] Machine Learning - WAYR (What Are You Reading) - Week 38**
    - Meta threads about papers read by users in the current week. Recommendations. Probably some useful papers there.
    - Data Science, DNN, Deep Learning, Deep Neural Network, CNN, RNN, Convolutional Neural Network, 
    Recurrent Neural Network, training, regularization, regularisation, hyperparameter optimization, NLP, tips, tipps, 
    hyperparameter optimisation, Natural Language Processing, text classification, classifier, gradient descent, ...
    - [ML subreddit week 38](https://www.reddit.com/r/MachineLearning/comments/7kgcqr/d_machine_learning_wayr_what_are_you_reading_week)
8. **Recurrent Highway Networks**
    - Apparently better then LSTMs. Implementation for Torch7 probably convertible to PyTorch.
    - RNN, Recurrent Neural Network, DNN, Deep Learning, Deep Neural Network, sota, state-of-the-art, 
    state of the art, code, maybe useful
    - [Paper at arvix](https://arxiv.org/abs/1607.03474)
    - [Torch & Tf implementation at Github I](https://github.com/julian121266/RecurrentHighwayNetworks); 
    [Torch & Tf implementation at Github II](https://github.com/BinbinBian/RecurrentHighwayNetworks); 
    [PyTorch Jupyter Notebook at Github](https://gist.github.com/a-rodin/d4f2ab5d7eb9d9887b26f28144e4ffdf#file-robinson-ipynb)
9. **SeLU activation aka Self-Normalizing Neural Networks**
    - Can work better then ReLU, has self-regularising properties, combined with alpha-dropout. Included in PyTorch.
    - scaled exponential linear unit, activation function, non-linearity, sota, state-of-the-art, state of the art, 
    non linearity, self-normalising, regularisation, regularization, feedforward nets, feedforward neural networks, 
    alpha dropout, feed-forward nets, feed-forward neural networks
    - [paper at arvix](https://arxiv.org/abs/1706.02515)
    - [PyTorch SeLU](http://pytorch.org/docs/0.3.0/nn.html#selu); 
    [PyTorch a-Dropout](http://pytorch.org/docs/0.3.0/nn.html#alphadropout)
    [ML subreddit discussion about self-normalising activations](https://www.reddit.com/r/MachineLearning/comments/6g5tg1/r_selfnormalizing_neural_networks_improved_elu)
10. **Automatic Sarcasm Detection: A Survey**
    - Overview over different works on sarcasm detection - until ~06/2017
    - Natural Language Processing, NLP, Deep Learning, Machine Learning, DNN, Deep Neural Networks
    - [Paper at institut](https://www.cse.iitb.ac.in/~pb/papers/acm-csur17-sarcasm-survey.pdf)
    - [Paper at arvix](https://arxiv.org/abs/1602.03426v1)                       
11. **Modelling Context with User Embeddings for Sarcasm Detection in Social Media**
    - July 2016. By contrast, we propose to automatically learn and then exploit user embeddings, 
    to be used in concert with lexical signals to recognize sarcasm. Does not seem to be useful for us.
    - Natural Language Processing, NLP, Deep Learning, Machine Learning, DNN, Deep Neural Networks
    - [Paper at arvix](https://arxiv.org/abs/1607.00976)                 
12. **Are Word Embedding-based Features Useful for Sarcasm Detection?**
    - Oct 2016. We explore if prior work can be enhanced using semantic similarity/discordance between word embeddings. 
    We augment word embedding-based features to four feature sets reported in the past. 
    - Natural Language Processing, NLP, Deep Learning, Machine Learning, DNN, Deep Neural Networks
    - [Paper at arvix](https://arxiv.org/abs/1610.00883)                 
13. **A Large Self-Annotated Corpus for Sarcasm**
    - Apr 2017. Paper about **our corpus** with some baselines.
    - Sarcasm detection, sarc, baseline, dataset, data set, data-set, text corpus, text-corpus
    - [Paper at arvix](https://arxiv.org/abs/1704.05579)
    - [Eval code for SARC @Github](https://github.com/NLPrinceton/SARC); 
    [ML subreddit - empty](https://www.reddit.com/r/textdatamining/comments/6h0cbr/a_large_selfannotated_corpus_for_sarcasm)           
14. **Creating and Characterizing a Diverse Corpus of Sarcasm in Dialogue**
    - To demonstrate the properties and quality on our  corpus,   we  conduct  supervised learning 
    experiments with simple features, and  show  that  we  achieve  both  higher precision  and  F 
    than  previous  work  on sarcasm  in  debate  forums  dialogue
    - Natural Language Processing, NLP, SVM, Support Vector Machine, No Deep Learning, alternative classifier
    - [Paper at aclweb](http://aclweb.org/anthology/W/W16/W16-3604.pdf)
15. **Attentive Convolution**
    - deriving higher-level features for a word not only from local context, 
    but also from information extracted from nonlocal context by the attention mechanism
    - CNN, convolutional neural networks, state-of-the-art, sota, state of the art, DNN, Deep Learning,
    Deep Neural Network, text classification, convolution with attention
    - [Paper at arvix](https://arxiv.org/abs/1710.00519)
    - [Implementation at Github](https://github.com/yinwenpeng/Attentive_Convolution); 
    - [Keras Implementation at Github](https://gist.github.com/stoney95/9c71bb7f7008b7d01dcf8f6bf3afaeff)
16. **A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification**
    - We focus on one-layer CNNs [..] We derive practical advice from our extensive empirical results
    - CNN, convolutional neural networks, Deep Neural Network, text classification, DNN, Deep Learning
    - [Paper at arvix](https://arxiv.org/abs/1510.03820)

Other papers
---------------------------------------
1. **Title**
    - Summary
    - Keywords
    - [hyperlink]()
    - [relevant]()

Data Science discussions and questions
---------------------------------------
1. **Data augmentation in NLP, text features**
    - Preprocessing, Natural Language Processing, enrichment, augmentation
    - [fast ai discussion](http://forums.fast.ai/t/data-augmentation-for-nlp/229)
2. **Max-over-time pooling vs no max-pooling for text classification?**
    - different word lengths, sum pooling, mean pooling, attention, last RNN state, max-out layer
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7m772v/d_maxovertime_pooling_vs_no_maxpooling_for_text)
    - [Max-over-time pooling paper I](http://arxiv.org/abs/1408.5882); 
    [Max-over-time pooling paper II](https://arxiv.org/abs/1103.0398); 
    [No-max-poling Blog](https://towardsdatascience.com/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575)
3. **Title**
    - Summary
    - Keywords
    - [hyperlink]()
    - [relevant]()
    
Machine Learning discussions and questions
---------------------------------------
1. **Dropout discussion:**
    - In case of RNNs tie the dropout masks across time. In case of CNN's: Follow hyperlink
    - Drop out, Drop-out, Deep Learning, Regularisation, Regularization, Recurrent, Neural Networks, CNN, 
    Convolutional Neural Networks, Recurrent Neural Networks
    - https://arxiv.org/abs/1512.05287
    - https://www.reddit.com/r/MachineLearning/comments/5l3f1c/d_what_happened_to_dropout/
2. **Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models**
    - batch-norm, Batch Norm, Batchnorm, Batch-Renorm, Batch Renorm, Batch Renormalization, 
Batchrenorm, normalization, normalisation, Batch-Renormalization, Batch-Normalization
    - [ML subreddit I](https://www.reddit.com/r/MachineLearning/comments/5tr0cd/r_batch_renormalization_towards_reducing); 
    [NL subreddit II](https://www.reddit.com/r/MachineLearning/comments/7issby/d_training_with_batch_normalization)
    - [arvix Batch Renorm paper](https://arxiv.org/abs/1702.03275); 
    [Batch norm paper at arvix](https://arxiv.org/abs/1502.03167)
3. **\[D\] What heuristics / rule of thumb / discoveries have you made during your work on machine learning**
    - Machine Learning experiences
    - Deep Learning, DNN, Deep Neural Network, training, overfitting, practice, tips, tricks
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/78u5k6/d_what_heuristics_rule_of_thumb_discoveries_have)
4. **Questions about Convolutional RNNs**
    - Links to papers inside. Unofficial PyTorch implementations available.
    - Convolutional Recurrent Neural Networks, CRNN, CNN + RNN, DNN, Deep Learning, Machine Learning, 
    Deep Neural Networks, Convolutional Neural Networks, convolution, convolutional filter
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/4waw0g/questions_about_convolutional_rnns)
    - [PyTorch implementation I](https://github.com/bgshih/crnn); 
    [PyTorch implementation II](https://github.com/meijieru/crnn.pytorch)
  
Research Question Pascal
---------------------------------------
1. **Gradient Boosting from scratch**
    - Gradient Boosted Models tutorial
    - GBM, Ensemble, Bagging, Boosting, 
    - [Medium blog](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
2. **A Kaggle Master Explains Gradient Boosting**
    - GBM, Boosting, XGBoost, XG-Boost, Gradient Boosted Models,
    - [Kaggle tutorial blog](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting)
3. **Extremely randomized trees**
    - It essentially consists of randomizing strongly both attribute and cut-point choice 
    while splitting a tree node. In the extreme case, it builds totally randomized trees 
    whose structures are independent of the output values of the learning sample.
    - Random Forests, classifier, classification
    - [Paper at Springer](https://link.springer.com/article/10.1007%2Fs10994-006-6226-1); 
    [Paper at semanticscholar](https://pdfs.semanticscholar.org/336a/165c17c9c56160d332b9f4a2b403fccbdbfb.pdf)
    - [scikit-learn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier); 
    [ML subreddit discussion](https://www.reddit.com/r/MachineLearning/comments/29uwi1/why_are_extremely_randomized_trees_more_efficient)
4. **CatBoost**
    - Gradient Boosting on decision trees library
    - GBM, state-of-the-art, sota, state of the art, very fast, runs on GPU, GPU
    - [Repos at Github](https://github.com/catboost)
    - [Blog of developers](https://catboost.yandex/news#version_0_6)
    
Research Question Gabriel
---------------------------------------
1. **Partial Information Attacks on Real-world AI**
    - Summary
    - Adversarial attack, Adversarial training, Deep Neural Network, Blackbox, Black-box, Black box
    - [Labsix blog post](http://www.labsix.org/partial-information-adversarial-examples); 
    [Paper](http://www.labsix.org/papers/#blackbox)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7l2oxj/r_partial_information_attacks_on_realworld_ai)
2. **HotFlip: White-Box Adversarial Examples for NLP**
    - We propose an efficient method to generate white-box adversarial examples 
    that trick character-level and word-level neural models
    - Adversarial attack, Adversarial training, Deep Neural Network, whitebox, white box, 
    Natural Language Processing
    - [Paper at arvix](https://arxiv.org/abs/1712.06751)
    - [ML subreddit - yet emtpy](https://www.reddit.com/r/LanguageTechnology/comments/7mcgkj/hotflip_whitebox_adversarial_examples_for_nlp_pdf)
    
Frameworks, software and data
---------------------------------------
1. **PyTorch**
    - Deep Learning framework, DNN, Deep Neural Networks, code, software, CUDA, fast, sota, state-of-the-art, 
    state of the art
    - [PyTorch homepage](http://pytorch.org); [PyTorch at Github](https://github.com/pytorch); 
    [PyTorch 0.3 documentation](http://pytorch.org/docs/0.3.0/)
    - [PyTorch subreddit](https://www.reddit.com/r/pytorch)
2. **PyTorch tutorial**
    - See also *PyTorch tutorials from MILA*
    - Video tutorial, presentation, slides, Deep Learning Framework, introduction
    - [Youtube playlist](https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&disable_polymer=true)
    - [Google drive docs](https://drive.google.com/drive/folders/0B41Zbb4c8HVyUndGdGdJSXd5d3M)
3. **\[P\] PyTorch tutorials from MILA**
    -  See also *PyTorch tutorial*
    - Jupyter Notebook, Python, Deep Learning Framework, introduction.
    - [Github](https://github.com/mila-udem/welcome_tutorials/tree/master/pytorch)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/7cw3ts/p_pytorch_tutorials_from_mila)
4. **Gensim**
    - Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
    Install with Pip: gensim
    - NLP, Natural Language Processing, preprocessing, pre-processing, Information Retrieval, IR, framework
    - [Software at Github](https://github.com/RaRe-Technologies/gensim)
    - [Framework homepage](https://radimrehurek.com/gensim/index.html)
5. **Hyperas & Hyperopt**
    - *Still compatible???* A very simple convenience wrapper around hyperopt for fast prototyping with keras models.
     \[..\] Instead, just define your keras model as you are used to, but use a simple template notation to 
     define hyper-parameter ranges to tune.
    - Keras wrapper, hyperparameter optimization, hyperparameter optimisation, grid search, library
    - [Implementation at Github](https://github.com/maxpumperla/hyperas)
    - [ML subreddit](https://www.reddit.com/r/MachineLearning/comments/46myc4/hyperas_keras_hyperopt_a_very_simple_wrapper_for/?submit_url=https%3A%2F%2Fgithub.com%2Fmaxpumperla%2Fhyperas&already_submitted=true&submit_title); 
    [Hyperopt - the thing which has been wrapped around](https://github.com/hyperopt/hyperopt)
6. **spaCy**
    - spaCy v2.0 excels at large-scale information extraction tasks. It's written from the ground up in 
    carefully memory-managed Cython. Huge collection of preprocessing tools. Can also use Gensim. 
    - Named Entity recognition, NLP, Natural Language Processing, pre-processing, POS, part-of-speech-tagging, 
    part of speed tagging, word-vectors, word vectors, word-embeddings, word embeddings, pipeline, sota,
    state-of-the-art, state of the art, tokenization, framework
    - [spaCy homepage](https://spacy.io)
    - [Blog about spaCy & co.](https://explosion.ai/blog); 
    [Blog entry about spaCy v1.0 & Keras](https://explosion.ai/blog/spacy-deep-learning-keras); 
    [Tutorial at youtube](https://www.youtube.com/watch?v=6zm9NC9uRkk); 
    [ML subreddit about tutorial](https://www.reddit.com/r/MachineLearning/comments/64m8ru/p_modern_nlp_in_python_what_you_can_learn_about); 
    [textacy, spaCy Wrapper](https://textacy.readthedocs.io/en/latest)
7. **GloVe**
    - Includes pretrained word vectors, code and paper
    - word-vectors, word embeddings, word-embeddings, wordvectors, preprocessing, pre-processing, data
    - [GloVe homepage at Stanford](http://nlp.stanford.edu/projects/glove)
8. **scikit-learn**
    - Simple and efficient tools for data mining and data analysis. Slow! Grid search doesn't allow
    for extra preprocessing for every cross validation fold. Alternatives: spaCy, Gensim and H2O  
    - framework, Machine Learning, ensemble, algorithms, Random Forest, SVM, Extra trees, linear regression, 
    logistic regression, cross-validation, supervised learning, unsupervised learning, classification, 
    classifiers, preprocessing, pre-processing, clustering, k-Means, KNN, K-Nearest Neighbours, Dimensionality
    reduction, dimension reduction, PCA, ...
    - [scikit-learn homepage](http://scikit-learn.org/stable)
    - [Video tutorials at Kaggle](http://blog.kaggle.com/?s=scikit-learn)
9. **H2O**
    - Comparable to scikit-learn, but apparently considerably faster.
    - framework, Machine Learning, ensemble, algorithms, Random Forest, linear regression, h2o.ai, h20.ai, 
    logistic regression, supervised learning, unsupervised learning, classification, cross-validation, 
    classifiers, clustering, k-Means, Dimensionality reduction, dimension reduction, PCA, Gradient Boosting, ...
    - [H2O homepage](https://www.h2o.ai/h2o)
    - [Documentation](http://docs.h2o.ai/h2o/latest-stable/index.html)   
10. **Tensorboard-PyTorch**
     - Tensorflows Tensorboard for PyTorch (Visualisation).
     Pip: tensorboardX, tensorflow
     - visualization, framework, software, training
     - [implementation at Github](https://github.com/lanpa/tensorboard-pytorch)
     - [Documentation at readthedocs](https://readthedocs.org/projects/tensorboard-pytorch); 
     [Short example at erogol blog](http://www.erogol.com/use-tensorboard-pytorch)
11. **Advances in Pre-Training Distributed Word Representations**
    - New state of the art word vectors that combine several techniques. 
    The pretrained ones are only good in case of English, because of the far larger corpus.
    Compare them nevertheless with others like GloVe, if possible. 
    **CAUTION!** Consider the fact, that we are dealing with sarcasm. Words suddenly have a different meaning.
    - word-vectors, wordvectors, word-embeddings, word embeddings, state-of-the-art, sota, preprocessing,
    NLP, Natural Language Processing, pre-processing
    - [files at fasttext.cc](https://fasttext.cc/docs/en/english-vectors.html)
    - [Explaining paper at arvix](https://arxiv.org/abs/1712.09405); 
    [ML subreddit discussion](https://www.reddit.com/r/MachineLearning/comments/7mtqxz/r_new_fasttext_paper_advances_in_pretraining)
12. **Torchtext**
    - Library for text preprocessing, batching, word-vector handling for PyTorch
    - pre-processing, word-vectors, wordvector, word vector, word-embedding, word embedding, embedding layers, 
    embedding-layers, batch-handling, batch handling, PyTorch extension 
    - [Torchtext at Github](https://github.com/pytorch/text)
    - [Tutorial at Blog](http://anie.me/On-Torchtext/); 
    [Reverse Tokenization at Github](https://github.com/jekbradbury/revtok)
13. **Char n-gram vectors**
    - word-vectors, wordvectors, word-embeddings, word embeddings, state-of-the-art, sota, preprocessing,
    NLP, Natural Language Processing, pre-processing
    - [](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/)
    - [Implementation at Github](https://github.com/hassyGo/charNgram2vec); 
    [Unfinished PyTorch version at Github](https://github.com/hassyGo/pytorch-playground/tree/master/jmt);
    [SQLite for word vectors](https://github.com/vzhong/embeddings)
14. **Tensor Comprehensions in PyTorch**
    - faster Cuda cuda, faster DNN code, faster Deep Neural Network code, faster NN code
    , faster Neural Network code, faster layer, faster NN layer, faster Neural Network layer
    , code by Evolutional Algorithm, code by Evolutionary Algorithm, Pytorch
    - [PyTorch Blog entry](http://pytorch.org/2018/03/05/tensor-comprehensions.html)
    - [FB original blog post about Tensor Comprehensions](https://research.fb.com/announcing-tensor-comprehensions/)
  
Other resources
---------------------------------------
1. **Title**
    - Summary
    - Keywords
    - [hyperlink]()
    - [relevant]()
