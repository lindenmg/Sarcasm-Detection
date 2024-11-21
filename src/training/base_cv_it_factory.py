from src.preprocessing.datahandler import DataHandler
from src.training.abstract_cv_iterator_factory import AbstractCVIteratorFactory
from src.training.cv_split_index_container import CVSplitIndexContainer


class BaseCvIteratorFactory(AbstractCVIteratorFactory):
    posts = None
    replies = None
    labels = None

    def create_cv_split_indices(self, datadict):
        reply_idx = DataHandler.cv_train_val_indices(
            datadict['replies'], pairs=True, split_ratio=self.split_ratio)
        post_idx = DataHandler.conv_cv_idx_to_single(reply_idx)
        return CVSplitIndexContainer({
            'replies': reply_idx,
            'posts': post_idx,
            'labels': reply_idx
        })
