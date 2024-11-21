import importlib
from abc import ABC, abstractmethod

import src.tools.helpers as helpers
from src.training.cv_iterator import CVIterator


class AbstractCVIteratorFactory(ABC):
    """
    An abstract factory class for creating CVIterator (see documentation for
    CVIterator) objects for cross validation. The user has to create a subclass
    of AbstractCVIteratorFactory and implement his own implementation of the
    abstract method 'create_cv_split_indices'. By doing so, a CVIterator can be
    created with 'create_cv_iterator'

    Example
    -------
    >>> from src.strategies.dummy.dummy_cv_iterator_factory import DummyCVIteratorFactory
    >>> from sklearn.datasets import make_classification
    >>>
    >>> data, labels = make_classification()
    >>> datadict = {'data': data, 'labels': labels}
    >>>
    >>> cv_it_factory = DummyCVIteratorFactory(dataset_module=DummyFFN,
    >>>                                     dataset_classname='DummyDataset', split_ratio=0.9)
    >>> cv_it = cv_it_factory.create_cv_iterator(datadict)
    >>>
    >>> for fold, (train_dataloader, val_dataloader) in enumerate(cv_it):
    >>>     # Now that you have dataloaders for training and validation in every fold
    >>>     # of the cross validation, you can do your magic.
    >>>     pass
    """

    def __init__(self, dataset_module, dataset_classname, n_splits=None, split_ratio=0.9
                 , batch_size=1000, shuffle=False, pin_memory=False, batch_sampler_class=None
                 , batch_sampler_module=None, sampler_class=None, sampler_module=None
                 , best_batch_size=False, train_data_amount=196527, batch_search_window=3
                 , reply_lengths=None):
        """
        The parameters define how the resulting dataloaders will be initialized.

        CAUTION: Does currently not support alternative Samplers and BatchSamplers
        for the DataLoader apart from BucketRandomSampler and LazyBatchSampler!

        Parameters
        ----------
        dataset_module: str
            the module to for the Dataset class, which shall be used in the cross-validation.
        dataset_classname: str
            the class-name for the Dataset class, which shall be used in the cross-validation.
        n_splits: int
            the number of splits, that shall be created by the resulting CVIterator.
            If 'None', the maximum number of splits will be used.
        split_ratio: float
            the ratio for splitting the datasets in training- and validation-set.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle: bool
            set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        pin_memory: bool
            If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        batch_sampler_class: str
            Class that provides indices for the current batches
        batch_sampler_module: str
            Module hierarchy of the torch.utils.data.sampler.* class
        sampler_class: str
            Class that prepares data for batch selection
        sampler_module: str
            Module hierarchy of the torch.utils.data.sampler.* class
        best_batch_size: bool
            True, if you want to have the best fitting batch size calculated
            with helpers.get_best_batch_size, so that the last batch is almost
            as big as the other ones.
        train_data_amount: int
            The number of training examples in the train data set
            (complete data before cross-validation split)
        batch_search_window: int
            Multiplier of ``batch_size``
            The window in which is searched for examples of similar length
            Higher values lead to more equality but less randomness
        reply_lengths: torch.LongTensor
            Contains the length for each reply in ``data_dict``
        See Also
        --------
        https:///pytorch.org/docs/0.3.1/data.html?highlight=dataloader#torch.utils.data.DataLoader
        """

        # init members to avoid annoying warnings
        self.datadict = None
        self.cv_split_indices = None

        self.n_splits = n_splits
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.sampler = None
        self.batch_sampler = None
        self.batch_search_window = None
        self.reply_lengths = None

        import_ = importlib.import_module
        self.dataset_class = getattr(import_(dataset_module)
                                     , dataset_classname)

        if sampler_class is not None and sampler_module is not None:
            self.sampler = getattr(import_(sampler_module)
                                   , sampler_class)
            self.batch_search_window = batch_search_window
            self.reply_lengths = reply_lengths
        if batch_sampler_class is not None and batch_sampler_module is not None:
            self.batch_sampler = getattr(import_(batch_sampler_module)
                                         , batch_sampler_class)
        if best_batch_size:
            data_length = int(round(train_data_amount * split_ratio))
            data_length = data_length - data_length % 2
            batch_size, _ = helpers.get_best_batch_size(data_length, batch_size, residual=0.9)
            self.batch_size = int(batch_size)

    @abstractmethod
    def create_cv_split_indices(self, datadict):
        """
        Must be overwritten by the user. Method shall provide a container for the indices,
        with which the data can be split for the cross validation. The keys for
        the created CVSplitIndexContainer must be equal to the keys
            in 'datadict'.
        Returns
        -------
        CVSplitIndexContainer
            provides the split indices for every dataset in the parameter 'datadict'.
            The keys for the created CVSplitIndexContainer must be equal to the keys
            in 'datadict'.
            For more information, see the documentation for CVSplitIndexContainer
        """
        raise NotImplementedError

    def create_cv_iterator(self, datadict):
        """
        Creates a CVIterator object. The CVIterator will apply the user defined indices
        from 'self.create_cv_split_indices' On the 'datadict' parameter.
        Parameters
        ----------
        datadict: dict
            dict of datasets. The keys must be equal to the init-parameters in referenced
            dataset (as specified 'dataset_module' and 'dataset_classname').
        """
        self.datadict = datadict
        self.cv_split_indices = self.create_cv_split_indices(datadict)
        return CVIterator(self.datadict, self.cv_split_indices, self.dataset_class
                          , self.n_splits, self.split_ratio, self.batch_size
                          , self.shuffle, self.pin_memory, self.batch_sampler
                          , self.sampler, self.batch_search_window, self.reply_lengths)
