from src.training.cv_split_index_container import CVSplitIndexContainer
import pytest

idx_posts = [
    [[1, 2], [3]],
    [[1, 3], [2]],
    [[2, 3], [1]]]
idx_replies = [
    [[1, 2, 3, 4], [5, 6]],
    [[1, 2, 5, 6], [3, 4]],
    [[3, 4, 5, 6], [1, 2]]]

idx_labels = idx_replies
indices = {
    'posts': idx_posts,
    'replies': idx_replies,
    'labels': idx_labels
}


class TestCVSplitIdxContainer:

    def test_init(self):
        with pytest.raises(ValueError) as val_err:
            CVSplitIndexContainer({'replies': idx_replies, 'WAT': [0]})
        assert val_err.value.args[0] == 'All indices must have the same length. For questions ask Pascal'

    def test_len(self):
        cv_splits_idx = CVSplitIndexContainer(indices)
        assert len(cv_splits_idx) == len(idx_posts)

    def test_item(self):
        cv_splits_idx = CVSplitIndexContainer(indices)
        assert cv_splits_idx[0]['posts'] == idx_posts[0]
        assert cv_splits_idx[0]['replies'] == idx_replies[0]
        assert cv_splits_idx[0]['labels'] == idx_labels[0]

        assert cv_splits_idx[1]['posts'] == idx_posts[1]
        assert cv_splits_idx[1]['replies'] == idx_replies[1]
        assert cv_splits_idx[1]['labels'] == idx_labels[1]

        assert cv_splits_idx[2]['posts'] == idx_posts[2]
        assert cv_splits_idx[2]['replies'] == idx_replies[2]
        assert cv_splits_idx[2]['labels'] == idx_labels[2]
