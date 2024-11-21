from sklearn.datasets import make_classification

from src.tools.config import Config
from src.strategies.dummy.dummy_cv_iterator_factory import DummyCVIteratorFactory


class TestAbstractCVIteratorFactory:
    """
    This test ensures that the following classes are correctly implemented:
    * AbstractCVIteratorFactory
    * CVIterator
    As AbstractCVIteratorFactory needs to be subclassed correctly by the user,
    it is still possible that bugs are introduced by the user.
    """

    def test_iter(self):
        """
        * Tests if the parameters are correctly provided to the dataloaders
        * Tests if the data is correctly split in every fold
        """
        data, labels = make_classification()
        datadict = {'data': data, 'labels': labels}

        cv_it_factory = DummyCVIteratorFactory(dataset_module='src.strategies.dummy.dummy_dataset',
                                               dataset_classname='DummyDataset', split_ratio=0.9,
                                               batch_size=10, shuffle=False,
                                               pin_memory=False)
        cv_it = cv_it_factory.create_cv_iterator(datadict)

        cv_split_indices = cv_it_factory.cv_split_indices

        for fold, (train_dl, val_dl) in enumerate(cv_it):
            assert train_dl.batch_size == 10
            assert train_dl.num_workers == Config.hardware.n_cpu
            assert not train_dl.pin_memory

            train_label_indices = cv_split_indices[fold]['labels'][0]
            val_label_indices = cv_split_indices[fold]['labels'][1]
            train_data_indices = cv_split_indices[fold]['data'][0]
            val_data_indices = cv_split_indices[fold]['data'][1]

            assert (val_dl.dataset.labels == datadict['labels'][val_label_indices]).all()
            assert (val_dl.dataset.data == datadict['data'][val_data_indices]).all()
            assert (train_dl.dataset.labels == datadict['labels'][train_label_indices]).all()
            assert (train_dl.dataset.data == datadict['data'][train_data_indices]).all()
