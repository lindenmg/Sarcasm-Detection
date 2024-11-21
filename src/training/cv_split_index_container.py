class CVSplitIndexContainer:
    """
    The indeces for every dataset, and for every fold of a subclass of CVIteratorFactory.
    Examples
    --------
    >>>    idx_posts = [
    >>>        [[1,2],[3]],
    >>>        [[1,3],[2]],
    >>>        [[2,3],[1]]]
    >>>    idx_replies = [
    >>>        [[1,2,3,4],[5,6]],
    >>>        [[1,2,5,6],[3,4]],
    >>>        [[3,4,5,6],[1,2]]]
    >>>
    >>>    idx_labels = idx_replies
    >>>    cvSplitIndices = CVSplitIndexContainer({
    >>>        'posts': idx_posts,
    >>>        'replies': idx_replies,
    >>>        'labels': idx_labels})

    For further examples check out test.pytest.training.cv_split_index_container
    """

    def __init__(self, indices={}):
        self.indices = indices
        if not all(len(list(indices.values())[0]) == len(el) for el in indices.values()):
            raise ValueError('All indices must have the same length. For questions ask Pascal')

    def __len__(self):
        k = list(self.indices.keys())[0]
        return len(self.indices[k])

    def __getitem__(self, item):
        return {
            k: v[item] for k, v in self.indices.items()
        }
