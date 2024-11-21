import importlib
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.autograd.variable import Variable
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from src.tools.helpers import hash_dict, save_json_to_disk
from src.training.metrics import Metrics


class LearningSession:

    def __init__(self, args):
        """
        Executes a training / testing session with predefined preprocessing & NN model

        Features:
        * Trains PyTorch models with predefined hyper-parameter
        * Tests PyTorch models with predefined hyper-parameter
        * All hyper-parameters are provided with a JSON. Almost no coding necessary
        * Can create logs, including, but not limited to Tensorboard
        * Caches the trained model parameters
        * Caches the pre-processing pre-processing for efficiency

        ----
        * It would be nice to cumulate the loss of every epoch, once for positive
        loss changes, and once for negative loss changes. This would provide a metric
        for overfitting (when the negative loss change is high)
        * make the lr_scheduler optional (like self.early_stopper)

        Parameters
        ----------
        args
        """

        # Init members to avoid warnings
        self.train_mode = False
        self.test_mode = False
        self.train_full = False
        self.model = None
        self.criterion = None
        self.early_stopper = None
        self.optimizer = None
        self.lr_scheduler = None
        self.data_factory = None
        self.logger = None
        self.cv_iterator_factory = None
        self.epochs = None
        self.session_tag = None
        self.cache_prefix = None
        self.word_vectors = None
        self.reply_lengths = None
        self.embedding_size = None
        self.seed = 1337
        self.cuda = False
        self._initial_state_dict = None
        self.train_full_loader = None
        self.test_full_loader = None
        #####################################################

        self.args = args
        self._set_attributes(**args['learning_session'])
        args_to_hash = deepcopy(args)
        args_to_hash['learning_session'].pop('mode', None)
        args_to_hash['learning_session'].pop('save_interval', None)
        args_to_hash['learning_session'].pop('mode_to_load', None)
        args_to_hash['learning_session'].pop('fold_to_load', None)
        args_to_hash.pop('logger', None)
        self._args_hash = hash_dict(args_to_hash)
        self._load_logger()
        self._load_criterion()
        self._load_early_stopper()
        self._load_data_factory()
        self.train_metrics = Metrics(args['metrics']['output_activation'])
        self.test_metrics = Metrics(args['metrics']['output_activation'])
        self.train_full_metrics = Metrics(args['metrics']['output_activation'])
        self.val_metrics = Metrics(args['metrics']['output_activation'])
        torch.manual_seed(self.seed)  # <== TORCH SEED I !!!

    #  _       _ _   _       _ _          _   _
    # (_)_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  ___
    # | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \/ __|
    # | | | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | \__ \
    # |_|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|___/
    #

    def _set_attributes(self, mode, epochs, session_tag, seed, mode_to_load
                        , save_interval, cache_folder, fold_to_load
                        , cache_prefix='fnn_fact', cnn_mode=False):
        self.epochs = epochs
        self.session_tag = session_tag
        self.seed = seed
        self.save_interval = save_interval
        self.cnn_mode = cnn_mode
        self.cache_prefix = cache_prefix
        self.cache_folder = cache_folder
        self.fold_to_load = fold_to_load
        self.mode_to_load = mode_to_load
        if mode.find('train') >= 0:
            self.train_mode = True
        if mode.find('test') >= 0:
            self.test_mode = True
        if mode.find('tr_full') >= 0:
            self.train_full = True
        if not (self.train_mode or self.test_mode or self.train_full):
            raise ValueError("Please provide the string 'train' to the mode parameter "
                             "to activate training-mode. Or provide the string 'test' "
                             "to activate the test-mode. Or provide the string 'tr_full' "
                             "to activate training on full train set. Or a combination. "
                             "Or ask Pascal")

    def _load_model_class(self, args):
        self._object_generator('model', **args)

    def _load_criterion(self):
        self._object_generator('criterion', **self.args['criterion'])

    def _load_optimizer(self):
        if self.cnn_mode:
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params_to_optimize = self.model.parameters()
        args = self._update_args('optimizer', {'params': params_to_optimize})
        self._object_generator('optimizer', **args)

    def _load_lr_scheduler(self):
        args = self._update_args('lr_scheduler', {'optimizer': self.optimizer})
        self._object_generator('lr_scheduler', **args)

    def _load_early_stopper(self):
        if 'early_stopper' in self.args:
            self._object_generator('early_stopper', **self.args['early_stopper'])

    def _load_data_factory(self):
        dict_ = {'train': self.train_mode,
                 'test': self.test_mode,
                 'cache_dir': self.cache_folder,
                 'session_tag': self.session_tag,
                 'cache_prefix': self.cache_prefix}
        args = self._update_args('data_factory', dict_)
        self._object_generator('data_factory', **args)

    def _load_cv_iterator_factory(self):
        args = self.args['cv_iterator_factory']
        if torch.cuda.is_available():
            args = self._update_args('cv_iterator_factory', {'pin_memory': True})
        self._object_generator('cv_iterator_factory', **args)

    def _load_logger(self):
        args = self.args['logger']
        self._object_generator('logger', **args)

    def _update_args(self, category, dictionary):
        d = deepcopy(self.args[category])
        d['args'].update(dictionary)
        return d

    def _object_generator(self, attribute, module, class_name, args):
        class_ = getattr(importlib.import_module(module), class_name)
        obj = class_(**args)
        setattr(self, attribute, obj)

    #  _                       _
    # | | ___  __ _ _ __ _ __ (_)_ __   __ _
    # | |/ _ \/ _` | '__| '_ \| | '_ \ / _` |
    # | |  __/ (_| | |  | | | | | | | | (_| |
    # |_|\___|\__,_|_|  |_| |_|_|_| |_|\__, |
    #                                  |___/
    #                               _
    #  _ __  _ __ ___   ___ ___  __| |_   _ _ __ ___  ___
    # | '_ \| '__/ _ \ / __/ _ \/ _` | | | | '__/ _ \/ __|
    # | |_) | | | (_) | (_|  __/ (_| | |_| | | |  __/\__ \
    # | .__/|_|  \___/ \___\___|\__,_|\__,_|_|  \___||___/
    # |_|
    #

    def run(self):
        """
        Starts the learning procedure.
        """
        self.logger.log_session_start(self.args)
        result_dict = self.data_factory.get_data()
        train_datadict = result_dict['train_data']
        test_datadict = result_dict['test_data']
        self.word_vectors = result_dict.get('word_vectors', None)
        embedding_size = result_dict.get('embedding_size', None)
        reply_lengths = result_dict.get('reply_lengths', None)

        if self.word_vectors is not None:
            self.args['model'] = self._update_args('model', {'pt_vectors': self.word_vectors})
        if embedding_size is not None:
            self.args['model'] = self._update_args('model', {'emb_num': embedding_size})
        if reply_lengths is not None:
            key = 'cv_iterator_factory'
            args = {'reply_lengths': reply_lengths}
            self.args['cv_iterator_factory'] = self._update_args(key, args)
        self._load_model_class(self.args['model'])
        self._load_cv_iterator_factory()
        self._load_optimizer()
        self._load_lr_scheduler()
        self._eventually_activate_cuda()
        if self.train_mode:
            self.logger.log_training_start()
            self._train_mode(train_datadict)
            self.logger.log_training_finished()
        if self.train_full:
            self.logger.log_training_start()
            self.train_full_loader = self._create_dataloader(train_datadict
                                                             , reply_lengths)
            self._train_full_and_cache_maybe()
            self.logger.log_training_finished()
        if self.test_mode:
            self.test_full_loader = self._create_dataloader(test_datadict, None)
            self.logger.log_testing_start()
            self._test_mode()
            self.logger.log_testing_finished()
        self.logger.log_session_finished()
        result_dict = {}
        if self.test_mode:
            result_dict['testing'] = self.test_metrics.get_summary()
        if self.train_mode:
            result_dict['training'] = self.train_metrics.get_summary()
            result_dict['validation'] = self.val_metrics.get_summary()
        if self.train_full:
            result_dict['full_training'] = self.train_full_metrics.get_summary()
        return result_dict

    def _train_mode(self, datadict):
        cv_it = self.cv_iterator_factory.create_cv_iterator(datadict)
        for fold, (dataloader_train, dataloader_val) in enumerate(cv_it, 1):
            torch.manual_seed(self.seed)
            self.logger.log_fold_start(fold)
            n_train_batches = self._get_n_batches(dataloader_train)
            n_val_batches = self._get_n_batches(dataloader_val)
            last_train_batch_size = self._calc_last_batch_size(dataloader_train)
            last_val_batch_size = self._calc_last_batch_size(dataloader_val)
            train_batch_size = dataloader_train.batch_sampler.batch_size
            val_batch_size = dataloader_val.batch_sampler.batch_size
            self._reset_model_parameters()
            self._reset_learning_rate()
            self._reset_early_stopper()
            for epoch in range(1, (self.epochs + 1)):
                if self._eventually_stop():
                    self.logger.log_training_aborted()
                    break
                self._train(dataloader_train, self.train_metrics)
                train_loss, train_acc = self.train_metrics.get_metrics(train_batch_size
                                                                       , n_train_batches
                                                                       , last_train_batch_size)
                self._validate(dataloader_val, self.val_metrics)
                val_loss, val_acc = self.val_metrics.get_metrics(val_batch_size
                                                                 , n_val_batches
                                                                 , last_val_batch_size)
                self._adjust_learning_rate(val_loss)
                self._update_early_stopper(val_loss, val_acc)
                lr_inferred = self.optimizer.param_groups[0]['lr']
                self.logger.log_epoch_finished(epoch, train_acc, train_loss
                                               , val_acc, val_loss, lr_inferred)
                self._eventually_cache_model(epoch, fold, "train")
            self.logger.log_fold_finished(fold)
            self.val_metrics.fold_finished()
            self.train_metrics.fold_finished()

    def _train_full_and_cache_maybe(self):
        torch.manual_seed(self.seed)
        batch_num = self._get_n_batches(self.train_full_loader)
        batch_size = self.train_full_loader.batch_sampler.batch_size
        self._reset_model_parameters()
        self._reset_learning_rate()
        self._reset_early_stopper()
        self.logger.log_fold_start(1)
        epoch_range = self.epochs
        epoch_range += 1
        last_batch_size = self._calc_last_batch_size(self.train_full_loader)
        for epoch in range(1, epoch_range):
            if self._eventually_stop():
                self.logger.log_training_aborted()
                break
            self._train(self.train_full_loader, self.train_full_metrics)
            train_loss, train_acc = self.train_full_metrics.get_metrics(batch_size
                                                                        , batch_num
                                                                        , last_batch_size)
            self._adjust_learning_rate(train_loss)
            self._update_early_stopper(train_loss, train_acc)
            self.logger.log_epoch_finished(epoch, train_acc, train_loss, 0., 0.
                                           , self.optimizer.param_groups[0]['lr'])
            self._eventually_cache_model(epoch, 1, "tr_full")
        self.train_full_metrics.fold_finished()
        self.logger.log_fold_finished(1)

    def _test_mode(self):
        """
        Simple test mode. Tries to load saved model
        """
        model_temp = self.model
        model = self._load_saved_model(self.fold_to_load, self.mode_to_load)
        if model is not None:
            self.model = model
        elif not self.train_full:
            raise Warning("There is no model trained on the "
                          "full dataset to test with!")
        # Use NO self._reset_model_parameters() here !!!
        self._reset_learning_rate()
        self._reset_early_stopper()
        self.logger.log_fold_start(1)

        n_val_batches = self._get_n_batches(self.test_full_loader)
        val_batch_size = self.test_full_loader.batch_sampler.batch_size
        last_batch_size = self._calc_last_batch_size(self.test_full_loader)
        self._validate(self.test_full_loader, self.test_metrics)
        self.model = model_temp
        test_loss, test_acc = self.test_metrics.get_metrics(val_batch_size
                                                            , n_val_batches
                                                            , last_batch_size)
        lr_inferred = self.optimizer.param_groups[0]['lr']
        self.logger.log_epoch_finished(1, 0., 0., test_acc, test_loss, lr_inferred)
        self.test_metrics.fold_finished()
        self.logger.log_fold_finished(1)
        return test_loss, test_acc

    def _validate(self, dataloader, metrics):
        def _step(variable_dict, labels_):
            y_pred = self.model(**variable_dict)
            loss = self.criterion(y_pred, labels_)
            metrics.update(y_pred.data, labels_.data, loss.data)

        self.model.eval()
        if self.cuda:
            for data, labels in dataloader:
                _step(variable_dict={k: Variable(v, volatile=True).cuda()
                                     for k, v in data.items()}
                      , labels_=Variable(labels, volatile=True).cuda())
        else:
            for data, labels in dataloader:
                _step(variable_dict={k: Variable(v, volatile=True)
                                     for k, v in data.items()},
                      labels_=Variable(labels, volatile=True))

    def _train(self, dataloader, metrics):
        def _step(variable_dict, labels_):
            y_pred = self.model(**variable_dict)
            loss = self.criterion(y_pred, labels_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            metrics.update(y_pred.data, labels_.data, loss.data)

        # There goes the coding style for more speed
        # ... And yes, it is correct to NOT pass the CNN
        # with an embedding layer a PyTorch Variable (only for training)
        self.model.train()
        if self.cuda:
            if self.cnn_mode:
                for data, labels in dataloader:
                    _step(variable_dict={k: v.cuda()
                                         for k, v in data.items()},
                          labels_=Variable(labels, volatile=True).cuda())
            else:
                for data, labels in dataloader:
                    _step(variable_dict={k: Variable(v, requires_grad=True).cuda()
                                         for k, v in data.items()},
                          labels_=Variable(labels, volatile=True).cuda())
        else:
            if self.cnn_mode:
                for data, labels in dataloader:
                    _step(variable_dict={k: v
                                         for k, v in data.items()},
                          labels_=Variable(labels, volatile=True))
            else:
                for data, labels in dataloader:
                    _step(variable_dict={k: Variable(v, requires_grad=True)
                                         for k, v in data.items()},
                          labels_=Variable(labels, volatile=True))

    def _adjust_learning_rate(self, loss):
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    def _eventually_stop(self):
        if self.early_stopper is not None and self.early_stopper.should_stop():
            return True
        else:
            return False

    def _update_early_stopper(self, val_loss, val_acc):
        if self.early_stopper is not None:
            self.early_stopper.epoch_finished(val_loss, val_acc)

    def _reset_learning_rate(self):
        self._load_optimizer()
        self._load_lr_scheduler()

    def _reset_early_stopper(self):
        self._load_early_stopper()

    def _reset_model_parameters(self):
        if self._initial_state_dict is None:
            self._initial_state_dict = deepcopy(self.model.state_dict())
        else:
            self.model.load_state_dict(self._initial_state_dict)

    #  _          _
    # | |__   ___| |_ __   ___ _ __ ___
    # | '_ \ / _ \ | '_ \ / _ \ '__/ __|
    # | | | |  __/ | |_) |  __/ |  \__ \
    # |_| |_|\___|_| .__/ \___|_|  |___/
    #              |_|
    #

    def _create_dataloader(self, data_dict, reply_lengths):
        """This function was written in the hacked manner,
        it uses functions and attributes of other classes
        in an unusual manner"""
        dataset_class = self.cv_iterator_factory.dataset_class
        dataset = dataset_class(**data_dict)
        if torch.cuda.is_available():
            pin_memory = True
            batch_size = 1024
        else:
            pin_memory = False
            batch_size = 32
        if reply_lengths is not None:
            cv_iter = self.cv_iterator_factory.create_cv_iterator(data_dict)
            data_loader = cv_iter.get_dataloader(dataset, reply_lengths)
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4
                                     , pin_memory=pin_memory, drop_last=False)
        return data_loader

    def _eventually_activate_cuda(self):
        if torch.cuda.is_available():
            self.cuda = True
            self.criterion = self.criterion.cuda()
            self.model = self.model.cuda()

    @staticmethod
    def _get_n_batches(data_loader):
        return len(data_loader.batch_sampler)

    @staticmethod
    def _calc_last_batch_size(dataloader):
        batch_size = dataloader.batch_sampler.batch_size
        data_len = len(dataloader.dataset)
        reminder = data_len % batch_size
        return reminder

    #                 _     _
    #   ___ __ _  ___| |__ (_)_ __   __ _
    #  / __/ _` |/ __| '_ \| | '_ \ / _` |
    # | (_| (_| | (__| | | | | | | | (_| |
    #  \___\__,_|\___|_| |_|_|_| |_|\__, |
    #                               |___/

    def _eventually_cache_model(self, epoch, fold, mode):
        if self.cache_folder is not None \
                and self.save_interval is not None \
                and self.save_interval > 0 \
                and epoch % self.save_interval == 0:
            base_filename = self.session_tag + '_' + str(fold) \
                            + str(mode) + '_' + self._args_hash
            data_path = Path(self.cache_folder) / 'models' / 'data'

            json_path = Path(self.cache_folder) / 'models' / 'parameters'
            os.makedirs(str(json_path), exist_ok=True)
            json_path = json_path / (base_filename + '.json')
            model_dict = deepcopy(self.args['model'])
            if self.cnn_mode:
                model_dict["args"].pop("pt_vectors", None)
            save_json_to_disk(model_dict, str(json_path)
                              , ensure_ascii=True, sort_keys=True)

            os.makedirs(str(data_path), exist_ok=True)
            data_path = data_path / (base_filename + '.model')
            torch.save(self.model.state_dict(), str(data_path))

    def _load_saved_model(self, fold, mode):
        model = None
        if self.cache_folder is not None:
            base_filename = self.session_tag + '_' + str(fold) \
                            + str(mode) + '_' + self._args_hash
            data_path = Path(self.cache_folder) / 'models' / 'data'

            json_path = Path(self.cache_folder) / 'models' / 'parameters'
            json_path = json_path / (base_filename + '.json')
            data_path = data_path / (base_filename + '.model')
            if json_path.is_file() and data_path.is_file():
                with open(str(json_path)) as file:
                    model_dict = json.load(file)

                model_state_dict = torch.load(str(data_path))
                model_temp = self.model
                if self.word_vectors is not None:
                    model_dict["args"]['pt_vectors'] = self.word_vectors
                self._load_model_class(model_dict)
                model = self.model
                self.model = model_temp
                model.load_state_dict(model_state_dict)
                if self.cuda:
                    model.cuda()
        return model
