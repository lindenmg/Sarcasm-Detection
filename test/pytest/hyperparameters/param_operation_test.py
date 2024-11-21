from src.hyperparameters.param_operation import *
import numpy as np

# default_dict and tuple_dict are the values that might be passed
# to a subclass of AbstractParameterIterator
default_dict = {
    'param0': 'aloha',
    'param1': {
        'param10': 1,
        'param11': {
            'param110': 0
        },
        'param12': 'one'
    },
    'param2': 3,
    'param3': 0,
    'param4': 0,
    'param5': 10,
    'param6': [2, 2]
}

single_qualitative = {
    'keys': ['param0'],
    'val': ['aloha', 'servus']
}

single_qualitative_lists = {
    'keys': ['param6'],
    'val': [[1, 1], [100, 100], [2, 2]]
}

single_quantitative_discrete = {
    'keys': ['param5'],
    'min': 4,
    'max': 17,
    'quantity_type': 'int'
}

single_quantitative_continuous = {
    'keys': ['param2'],
    'min': 0.5,
    'max': 5.5,
    'quantity_type': 'float'
}

group_qualitative = {
    'keys': (['param1', 'param10'], ['param1', 'param12']),
    'val': ([1, 2, 3], ['one', 'two', 'three'])
}

group_quantitative = {
    'keys': (['param3'], ['param4']),
    'min': (0, 0),
    'max': (5, 100),
    'quantity_type': ('float', 'int')
}

variation_list = [single_qualitative, single_quantitative_continuous, single_quantitative_discrete,
                  group_qualitative, group_quantitative]

# This a param dict, as it might be sampled in a random search.
sampled_params_dict_01 = {
    'param0': 'servus',
    'param1': {
        'param10': 3,
        'param11': {
            'param110': 0
        },
        'param12': 'three'
    },
    'param2': 3.14159,
    'param3': 2.565,
    'param4': 51,
    'param5': 13,
    'param6': [2, 2]
}

params_vector_01 = [1, 2, 3.14159, 2.565, 13]
params_vector_02 = [1, 2, 3.14159, 2.565, 13.2]


class TestParamOperation:

    def test_flatten_complete(self):
        list_of_lists = ['hallo', 'string', 3.14159, ['nested', 23]]
        flat = flatten_complete(list_of_lists)
        assert flat == ['hallo', 'string', 3.14159, 'nested', 23]

    def test_fatten_dict_auto(self):
        dictionary = {
            'k0': 0,
            'k1': {
                'k10': 10,
                'k11': {
                    'k110': 110,
                    'k111': 111}}}

        result = flatten_dict_auto(dictionary, sep='.')
        expected_keys = ['k1.k11.k110', 'k1.k11.k111', 'k1.k10', 'k0']
        assert all([k in expected_keys for k in result.keys()])
        assert dictionary['k0'] == result['k0']
        assert dictionary['k1']['k10'] == result['k1.k10']
        assert dictionary['k1']['k11']['k110'] == result['k1.k11.k110']

    def test_flatten_dict(self):
        dictionary = {
            'k0': 0,
            'k1': {
                'k10': 10,
                'k11': {
                    'k110': 110,
                    'k111': 111}}}
        result = flatten_dict(dictionary, [('k0', 'k0_filtered'), ('k10', 'k10_filtered'), ('k110', 'k110_filtered')])
        assert dictionary['k0'] == result['k0_filtered']
        assert dictionary['k1']['k10'] == result['k10_filtered']
        assert dictionary['k1']['k11']['k110'] == result['k110_filtered']

    def test_traverse_get(self):
        v = traverse_get(sampled_params_dict_01, ['param1', 'param11', 'param110'])
        assert v == sampled_params_dict_01['param1']['param11']['param110']

    def test_sort_variation_list(self):
        sorted_list = sort_variation_list(variation_list)
        assert sorted_list[0]['keys'] == ['param0']
        assert sorted_list[1]['keys'] == (['param1', 'param10'], ['param1', 'param12'])
        assert sorted_list[2]['keys'] == ['param2']
        assert sorted_list[3]['keys'] == (['param3'], ['param4'])

    def test_params_dict_to_vector(self):
        converted_vector_0 = params_dict_to_vector(variation_list, sampled_params_dict_01)
        assert np.equal(converted_vector_0, params_vector_01).all()

        variation_list_1 = [single_qualitative, single_qualitative_lists]
        converted_vector_1 = params_dict_to_vector(variation_list_1, sampled_params_dict_01)
        assert converted_vector_1 == [1, 2]

    def test_params_vector_to_dict(self):
        params_dict = params_vector_to_dict(default_dict, variation_list, params_vector_02)
        assert params_dict == sampled_params_dict_01

    def test_compare_dict_formats(self):
        d_right = sampled_params_dict_01
        d_wrong = {
            'param0': 'servus',
            'param1': {
                'param10': 3,
                'param11': {
                    'param110': 0
                },
            },
            'param2': 3.14159,
            'param3': 2.565,
            'param4': 51,
            'param5': 13
        }
        assert same_dict_format(d_right, d_wrong) == False
        assert same_dict_format(d_wrong, d_right) == False
        assert same_dict_format(d_right, d_right) == True

    def test_filter_by_keyslists(self):
        keyslists = [
            ['param1', 'param11', 'param110'],
            ['param1', 'param12'],
            ['param3']

        ]
        expected_dict = {
            'param1': {
                'param11': {
                    'param110': 0
                },
                'param12': 'one'
            },
            'param3': 0
        }

        filtered_dict = filter_by_keys_lists(default_dict, keyslists)
        assert filtered_dict == expected_dict
