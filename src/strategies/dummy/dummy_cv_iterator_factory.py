from src.preprocessing.datahandler import DataHandler
from src.training.abstract_cv_iterator_factory import AbstractCVIteratorFactory
from src.training.cv_split_index_container import CVSplitIndexContainer


class DummyCVIteratorFactory(AbstractCVIteratorFactory):
    datadict = None

    def create_cv_split_indices(self, datadict):
        idx = DataHandler.cv_train_val_indices(datadict['data'], pairs=False, split_ratio=self.split_ratio)
        return CVSplitIndexContainer({
            'data': idx, 'labels': idx
        })
