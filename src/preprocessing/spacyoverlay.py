from itertools import starmap

import spacy
from spacy.language import Language
from spacy.tokens import Token


class SpacyOverlay:

    def __init__(self, model_type='en_core_web_md', pipeline_to_remove=list(['textcat'])):
        """
        This class offers some useful functions for a better SpaCy experience.

        Parameters
        ----------
        model_type: str
            The name of the model which SpaCy should load.
            You should have downloaded it beforehand.
            As of SpaCy 2.0 that would be 'en', 'en_core_web_sm',
            'en_core_web_md' and 'en_core_web_lg'
        pipeline_to_remove: list of str
            The names of the pipeline steps to disable.
            Keep the dependencies of the parts in mind!
        """
        self._model_type = model_type
        self._pipeline_to_remove = pipeline_to_remove
        self.nlp = None

    def get_nlp(self):
        """
        Returns
        -------
        spacy.lang.en.English
            A SpaCy language model that can do
            operations on the english language
        """
        if self.nlp is None:
            self.nlp = spacy.load(self._model_type, disable=self._pipeline_to_remove)
        return self.nlp

    @staticmethod
    def add_stop_word_def(stop_word_file=None, stop_words=None):
        """
        This will create a ._.is_stop method which recognises stop words better.

        You can redefine this method. BUT it is applied at runtime. So you have
        to set it just before you want to use it!
        Stop words are potentially useless words for text classification, etc.
        But be aware of the fact, that there might be some words which are
        significantly more frequent in one class than the other.

        Parameters
        ----------
        stop_word_file: str
            The path to a text file with stop words in it.
            Must have one word per line, no whitespaces apart from
            those of the line dividing type. They also have to be
            in lower case letter form.
        stop_words : set
            A set with the words you want to define as stop words.
            Higher order then stop_word_file. Gets chosen as long
            as it is not None
        """
        if stop_words and not isinstance(stop_words, set):
            stop_words = set(stop_words)

        if stop_word_file and stop_word_file != '':
            with open(stop_word_file, 'r') as f:
                stop_words = set(line.strip() for line in f)

        Token.set_extension('is_stop',
                            force=True,
                            getter=(lambda t: t.is_stop
                                              or (t.lower_ in stop_words)
                                              or (t.lemma_ in stop_words)))

    @staticmethod
    def load_word_vectors(words, vectors, nr_dim):
        """
        Creates a SpaCy Language object with custom vectors.

        CAUTION! It takes 20 minutes to create an object with 1M vectors!

        Parameters
        ----------
        words: list
            A list-like object with the words as strings
        vectors: np.ndarray
            A Numpy Array with the vector values as floats
        nr_dim: int
            The dimension of the word vectors in the file
        Returns
        -------
        spacy.language
            A SpaCy Language models which has the word vectors from
            the file in its vocabulary and NOTHING else.
        """
        nlp_vec = Language()
        nlp_vec.vocab.reset_vectors(width=nr_dim)
        set_vector = nlp_vec.vocab.set_vector
        result = starmap(set_vector, zip(words, vectors))
        return nlp_vec
