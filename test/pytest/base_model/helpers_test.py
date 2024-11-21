import pytest

from src.tools.helpers import *

dictionary = {
    'k0': 0,
    'k1': {
        'k10': 10,
        'k11': {
            'k110': 110,
            'k111': 111
        }
    }
}


class TestHelpers:

    def test_create_length_tensor_i(self):
        randint = np.random.randint
        test_iter = 20
        for _ in range(test_iter):
            test_list = []
            length_list = []
            cols = randint(0, 5)
            test_length = np.random.randint(100, 100000)

            for i in range(test_length):
                rand_length = randint(0, 30)
                shape = (rand_length, cols) if cols else (rand_length,)
                size = rand_length * cols if (cols > 1) else rand_length
                length_list.append(rand_length)
                array = np.random.permutation(size)
                array = np.reshape(array, shape) if (size and cols > 1) else array
                inner_list = array.tolist()
                test_list.append(inner_list)

            length_tensor = create_length_tensor(test_list)
            length_list = np.asarray(length_list, dtype=int)
            length_array = length_tensor.numpy().astype(int, copy=False)
            assert (length_list == length_array).all()

    def test_create_length_tensor_ii(self):
        array = np.random.permutation(100)
        message = "data_list should be list or tuple, " \
                  "but is {}".format(type(array))

        with pytest.raises(TypeError) as type_err:
            create_length_tensor(array)
        assert message in str(type_err.value)

    def test_get_best_batch_size_i(self):
        data_length = [1000, 1000, 1000, 1000, 1000, -1]
        batch_size = [100, 100, 100, 100, 10001, 100]
        residual = [1.001, -0.001, 0.5, 0.6, 0.8]
        window = [0.1, 0.3, -0.01, 1, 0.1, 0.2]
        for d, b, r, w in zip(data_length, batch_size, residual, window):
            with pytest.raises(ValueError) as val_err:
                get_best_batch_size(d, b, w, r)
            assert len(str(val_err.value)) > 0

    def test_get_best_batch_size_ii(self):
        data_length = [1001.3, 1000]
        batch_size = [100, 101.324]
        residual = [0.6, 0.8]
        window = [0.1, 0.2]
        for d, b, r, w in zip(data_length, batch_size, residual, window):
            with pytest.raises(TypeError) as type_err:
                get_best_batch_size(d, b, w, r)
            assert len(str(type_err.value)) > 0

    def test_get_best_batch_size_iii(self):
        iters = 1000
        data_length = np.random.randint(10000, 200000, iters).tolist()
        batch_size = np.random.randint(16, 2049, iters).tolist()
        residual = np.random.choice([None, 0.9, 0.8, 0.7], iters
                                    , p=[0.5, 0.2, 0.15, 0.15])
        residual = residual.tolist()
        for d, b, r in zip(data_length, batch_size, residual):
            best_batch, found = get_best_batch_size(d, b, residual=r)
            initial_rest = d % b
            best_rest = d % best_batch
            good_rest = True
            if r is not None:
                good_rest = best_rest >= (int(round((1 - r) * b)))
            assert initial_rest >= best_rest
            assert good_rest == found
