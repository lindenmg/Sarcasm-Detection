from src.tools.config import Config
import warnings
import os

testdata_path = filepath = os.path.realpath(os.path.join(__file__, "..", "..", "..", "test_data"))


class TestPreprocessing:

    # The test ensures that the default config (<project_root>/config.json) is loaded
    # and can be accessed with 'Config.config'
    def test_init(self):
        assert 'config_dict' in Config.__dict__.keys()

    # For loading a different config file
    def test_load_json(self):
        fpath = os.path.join(testdata_path, 'config_test.json')
        Config.load_json(fpath)
        assert Config.config_dict.get('p_1') == 42

    def test_automatic_member_injection(self):
        fpath = os.path.join(testdata_path, 'config_test.json')
        Config.load_json(fpath)
        assert Config.p_1 == 42
        assert Config.p_2 == 27
        assert Config.subconf_1.p_4 == 25
        assert Config.subconf_1.subconf_2.p_5 == 28
        assert 3 in Config.subconf_1.subconf_2.list_1

        raised = False
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                fpath = os.path.join(testdata_path, 'config_test_bad.json')
                Config.load_json(fpath)
            except Warning:
                raised = True
        assert raised

    def test_data_path(self):
        fpath = os.path.join(testdata_path, 'config_test.json')
        Config.load_json(fpath)
        assert Config.data_path("foo.txt") == "/the/path/to/the/data/folder/foo.txt"
