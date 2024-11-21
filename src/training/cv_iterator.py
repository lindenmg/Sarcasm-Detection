import torch
from torch.utils.data import DataLoader

from src.preprocessing.datahandler import DataHandler
from src.tools.config import Config


# The Factory has to be refactored too for that as well
class CVIterator:
    """
    Can be used to generate a training-dataloader and a validation-dataloader
    for every fold of a cross validation. The CVIterator is applied
    by src.training.learning_session.LearningSession. It is suggested to
    create a CVIterator with a user-defined subclass of AbstractCVIteratorFactory.
    """

    def __init__(self, datadict, cv_split_indices, dataset_class
                 , n_splits=10, split_ratio=0.9, batch_size=2
                 , shuffle=True, pin_memory=False, batch_sampler=None
                 , sampler=None, batch_search_window=None
                 , reply_lengths=None):
        """
        All necessary parameters for creating a cross validation iterator.

        Be aware that based on the fact how the PyTorch data-loader
        classes work, some parameters are mutually exclusive.
        Set the parameter X to None, if not compatible with Y.
        CAUTION: Does currently not support alternative Samplers and
        BatchSamplers for the DataLoader apart from BucketRandomSampler
        and LazyBatchSampler!

        Parameters
        ----------
        datadict: dict
            dict of datasets. The keys must be equal to the init-parameters in referenced
            dataset (as specified in Dataset_class).
        cv_split_indices: CVSplitIndexContainer
            provides the split indices for every dataset provided in 'datadict'.
            The keys must be the same as the init models of the Dataset
            which is referenced with 'dataset_module' and 'dataset_classname'
            For more information, see the documentation for CVSplitIndexContainer
        dataset_class: class
            A class reference to a subclass of torch.utils.data.dataset.Dataset.
            The iterator will create objects of this class with the given data from
            'datadict' and provide them to the respective dataloader in every fold.
        n_splits: int
            The number of validation / training sets the
            current network is trained and tested on
        split_ratio: float
            Split ratio of training and validation sets
        batch_size: int
        shuffle: bool
            True, if the data shall be shuffled before every epoch
        pin_memory: bool
            Faster Cuda execution
        batch_sampler: torch.utils.data.sampler.BatchSampler
            Class that provides indices for the current batches
        sampler: torch.utils.data.sampler.Sampler
            Class that prepares data for batch selection
        batch_search_window: int
            Multiplier of ``batch_size``
            The window in which is searched for examples of similar length
            Higher values lead to more equality but less randomness
        reply_lengths: torch.LongTensor
            Contains the length for each reply in ``data_dict``

        See Also
        --------
        https://pytorch.org/docs/0.3.1/data.html?highlight=dataloader#torch.utils.data.DataLoader
        """

        self.datadict = datadict
        self.cv_split_indices = cv_split_indices
        self.dataset_class = dataset_class
        self.n_splits = n_splits
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = Config.hardware.n_cpu
        self.pin_memory = pin_memory
        self.batch_window = batch_search_window
        self.reply_lengths = None

        if reply_lengths is not None:
            dh = DataHandler
            self.reply_lengths = dh.cv_train_val_indices(reply_lengths, pairs=True
                                                         , split_ratio=split_ratio)
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.cuda = torch.cuda.is_available()

    def get_dataloader(self, data, reply_lengths):
        if self.sampler is not None:
            if not isinstance(reply_lengths, torch.LongTensor):
                reply_lengths = torch.from_numpy(reply_lengths)
            sampler_ = self.sampler(data, reply_lengths, self.batch_size
                                    , self.batch_window, self.cuda)

        if self.batch_sampler is not None:
            if self.sampler is not None:
                batch_sampler_ = self.batch_sampler(sampler_, self.batch_size, drop_last=False)

        if self.batch_sampler is not None:
            dl = DataLoader(data, batch_sampler=batch_sampler_
                            , num_workers=self.num_workers
                            , pin_memory=self.pin_memory)
        elif self.sampler is not None:
            dl = DataLoader(dataset=data, batch_size=self.batch_size
                            , shuffle=False, num_workers=self.num_workers
                            , pin_memory=self.pin_memory, sampler=sampler_)
        else:
            dl = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=self.shuffle
                            , num_workers=self.num_workers, pin_memory=self.pin_memory)
        return dl

    def __iter__(self):
        if self.n_splits is not None and self.n_splits < len(self.cv_split_indices):
            folds = range(self.n_splits)
        else:
            folds = range(len(self.cv_split_indices))
        reply_length_fold = ['dummy0', 'dummy1']

        for fold in folds:
            if self.reply_lengths is not None:
                reply_length_fold = self.reply_lengths[fold]
            fold_indices = self.cv_split_indices[fold]
            train_data = {k: self.datadict[k][v[0].tolist()] for k, v in fold_indices.items()}
            eval_data = {k: self.datadict[k][v[1].tolist()] for k, v in fold_indices.items()}
            dataset_train = self.dataset_class(**train_data)
            dataset_eval = self.dataset_class(**eval_data)
            yield (self.get_dataloader(dataset_train, reply_length_fold[0])
                   , self.get_dataloader(dataset_eval, reply_length_fold[1]))
