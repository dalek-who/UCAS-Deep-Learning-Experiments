from typing import Tuple, Union

import torch
import torchvision
from torchvision.datasets import mnist
from torchvision import transforms
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, SequentialSampler, RandomSampler, Subset, Dataset
import re
import argparse
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from functools import wraps
import random
import os
import numpy as np
from tqdm import tqdm
import logging
import json
from itertools import chain
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from collections import defaultdict
from pathlib import Path
import shutil
from PIL import Image

from models.CNN import CNNModel
from models.AlexNet import AlexNet
from models.VGGNet import VGGNet
from config import ConfigTrain, ConfigFiles, ConfigModel, ConfigModel_VGG, ConfigModel_AlexNet
from utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images
from dataset import CatDogDataset

MODELS = {
    "CNNModel": CNNModel,
    "AlexNet": AlexNet,
    "VGGNet": VGGNet,
}

MODEL_CONFIGS = {
    "CNNModel": ConfigModel,
    "AlexNet": ConfigModel_AlexNet,
    "VGGNet": ConfigModel_VGG,
}

# 例外处理流程
def _wrap_handle_exception(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            return result
        except KeyboardInterrupt as e:
            self._handle_fail(keyboard_interrupt=True)
            raise
        except (Exception, SystemExit) as e:
            self._handle_fail(keyboard_interrupt=False)  # 例外处理流程
            raise
    return wrapped_func


class Experiment(object):
    def __init__(self):
        self.delete_log = True
        self.args = self._parse_args()
        self.DEBUG = self.args.debug
        self.ct = ConfigTrain()
        # self.cm = config.ConfigModel()
        ModelConfig = MODEL_CONFIGS[self.ct.model_name]
        self.cm = ModelConfig()
        self.cf = ConfigFiles(commit=self.args.commit, load_checkpoint_dir=self.args.load_checkpoint_dir, load_checkpoint_file=self.args.load_checkpoint_file, debug=self.DEBUG)

        self._set_seed()  # 设定随机种子必须在初始化model等所有步骤之前
        self.device: torch.device = self._get_device()

        self.iter_train, self.iter_train_eval, self.iter_valid, self.iter_test = self._get_data_iter()

        # self.writer = SummaryWriter(self.cf.DIR_OUTPUT)  # 这一步自动创建了DIR_OUTPUT
        self.writer = SummaryWriter(self.cf.DIR_TENSORBOARD)
        self.logger = init_logger(log_file=self.cf.out_log)
        self.best_eval_result = defaultdict(lambda: -1)

        self.model: torch.nn.Module = self._get_model(load_checkpoint_path=self.cf.load_checkpoint)
        assert self.model.__class__.__name__ == self.ct.model_name, (self.model.__class__.__name__, self.ct.model_name)
        self.model.to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=self.ct.learning_rate)
        # self.optimizer: optim.Optimizer = optim.SGD(self.model.parameters(), lr=self.ct.learning_rate, momentum=self.ct.momentum)
        assert self.optimizer.__class__.__name__ == self.ct.optimizer_name, (self.optimizer.__class__.__name__, self.ct.optimizer_name)
        # lr_lambda = lambda epoch: 1 / (1 + self.ct.lr_decay * epoch)
        # self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        # assert self.lr_scheduler.__class__.__name__ == self.ct.lr_scheduler_name, (self.lr_scheduler.__class__.__name__, self.ct.lr_scheduler_name)

        # 保存各种config
        with open(self.cf.out_config_files, "w") as f_cf, \
            open(self.cf.out_config_train, "w") as f_ct, \
            open(self.cf.out_config_model, "w") as f_cm, \
            open(self.cf.out_args, "w") as f_a:
            json.dump(vars(self.args), f_a, indent=4)
            json.dump(self.cf.to_jsonable_dict(), f_cf, indent=4)
            json.dump(self.ct.to_jsonable_dict(), f_ct, indent=4)
            json.dump(self.ct.to_jsonable_dict(), f_cm, indent=4)

    @_wrap_handle_exception
    def do_predict(self, iter_data) -> Tuple[np.ndarray, np.ndarray, float, int]:
        self.model.eval()
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        self.logger.info("***** Running evaluation/predict *****")
        label_true, label_pred = [], []  # 真实的tag id序列，预测的tag id序列，单词原文序列
        loss_total = 0.
        data_num = 0
        for bid, batch in enumerate(tqdm(iter_data, desc="Eval")):
            with torch.no_grad():
                data, label = [t.to(self.device) for t in (batch.img, batch.label)]
                pred_score = self.model(data)
                loss = F.cross_entropy(pred_score, label, reduction="sum")
                self.delete_log = False
            data_num += data.shape[0]
            loss_total += loss.item()
            label_true += label.tolist()
            label_pred += torch.argmax(pred_score, dim=1).tolist()
            # tags_raw_pred += [list(self.cm.vocab_tags.keys())[tag_id] for tag_id in chain.from_iterable(decode)]
        label_true, label_pred = np.array(label_true), np.array(label_pred)
        return label_true, label_pred, loss_total, data_num

    @_wrap_handle_exception
    def do_evaluation(self, iter_data) -> Tuple[dict, Subset, np.ndarray]:
        # Eval!
        label_true, label_pred, loss_total, data_num = self.do_predict(iter_data)
        metrics = self._metrics(y_true=label_true, y_pred=label_pred)
        error_subset, error_index = self._error_subset_and_index(dataset=iter_data.dataset, y_true=label_true,
                                                                 y_pred=label_pred)
        result: dict = {
            "loss_total": loss_total,
            "loss_mean": loss_total / data_num,
            "p": metrics["p"],
            "r": metrics["r"],
            "f1": metrics["f1"],
            "acc": metrics["acc"],
            "num_error": int(len(error_index)),
            "confusion_matrix": metrics["confusion_matrix"],
        }
        return result, error_subset, error_index

    @_wrap_handle_exception
    def do_train(self, iter_train, iter_train_eval, iter_valid, load_best_after_train=True):
        self.model.train()
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.logger.info("***** Running training *****")

        global_step = 0
        best_acc = 0
        self._set_seed()
        self.model.zero_grad()
        num_train_epochs = self.ct.num_train_epochs
        for epoch in tqdm(range(int(num_train_epochs)), desc="Train epoch"):
            for bid, batch in enumerate(tqdm(iter_train, desc="Train batch:")):
                data, label = [t.to(self.device) for t in (batch.img, batch.label)]
                pred_score = self.model(data)
                loss = F.cross_entropy(pred_score, label, reduction='mean')
                # loss = loss.mean() if self.args.n_gpu > 1 else loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ct.max_grad_norm)  # 梯度剪裁，把过大的梯度限定到固定值
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self._draw_each_step(global_step=global_step, loss=loss.item(), lr=lr) # 画图
                if bid==0:
                    self._draw_parameter_distribution(self.model, global_step=epoch, is_train=True, show_grad=True)  # 画梯度分布图
                self.optimizer.step()
                self.model.zero_grad()
                self.delete_log = False
                global_step += 1
            # self.scheduler.step()  # Update learning rate schedule
            eval_train, _, _ = self.do_evaluation(iter_train_eval)
            eval_valid, _, _ = self.do_evaluation(iter_valid)
            eval_result = dict(**{k+"_train": v for k,v in eval_train.items()},
                               **{k+"_valid": v for k,v in eval_valid.items()})
            eval_result["epoch"] = epoch
            self._draw_each_epoch(epoch=epoch, eval_result=eval_result, model=self.model)# 画图
            self._create_checkpoint(epoch=epoch, eval_result=eval_result)
        if load_best_after_train:
            if num_train_epochs>0:
                self.model = self._get_model(load_checkpoint_path=self.cf.out_checkpoint)
            self.logger.info(f"**********************  Load best model of epoch [{self.best_eval_result['epoch']}]  **********************")
        self._draw_best()
        # self.writer.export_scalars_to_json(self.cf.out_scalars)
        self.cf.out_success_train.open("w").write("Train Success!!")
        self.logger.info("*****  Train Success !!!  *****")

    @_wrap_handle_exception
    def analysis(self, iter_data):
        # 测试集上的效果
        eval_result, error_subset, error_index = self.do_evaluation(iter_data=iter_data)  # type: dict, Subset, np.ndarray
        eval_result_info = {k:v for k,v in eval_result.items() if isinstance(v, (int, float, str))}
        show_info = f'\nTest result: \n' + "\n".join([f' {key}: {value:.4f} ' for key, value in eval_result_info.items()])
        self.logger.info(show_info)

        # 展示超参数
        hparam_dict = dict(self.ct.to_jsonable_dict(), **self.cm.to_jsonable_dict())
        self._draw_hyper_param(hparam_dict=hparam_dict, metric_dict=eval_result_info)

        # 展示错误样本
        error_loader = DataLoader(dataset=error_subset, batch_size=64)
        error_batch = iter(error_loader).next()
        images, labels, names = error_batch
        self._draw_examples(images, self.cf.tbx_error_examples_test)
        # 保存错误样本的名字和index，便于查阅
        error_name_list = [example.name for example in error_subset]
        error_index_list = error_index.tolist()
        with open(self.cf.out_error_examples, "w") as f:
            json.dump({"name_list": error_name_list, "index_list": error_index_list}, f)


        # 展示各层参数的分布情况
        self._draw_parameter_distribution(self.model, is_test=True)
        pass



    def success(self):
        self.writer.close()
        if self.cf.DIR_OUTPUT.name.startswith("DEBUG-"):
            new_name = self.cf.DIR_OUTPUT.name.replace("DEBUG-", "DEBUG-success-")
        else:
            new_name = "success-" + self.cf.DIR_OUTPUT.name
        self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / new_name)
        self.logger.info("*****  Experiment Success  *****")

    def _handle_fail(self, keyboard_interrupt=False):
        self.writer.close()
        if self.delete_log and os.path.isdir(self.cf.DIR_OUTPUT):
            shutil.rmtree(self.cf.DIR_OUTPUT)
            return
        if self.model.training:
            self._create_checkpoint(epoch=-1, eval_result=None, fail=True)
        if not keyboard_interrupt:
            if self.cf.DIR_OUTPUT.name.startswith("DEBUG-"):
                new_name = self.cf.DIR_OUTPUT.name.replace("DEBUG-", "DEBUG-fail-")
            else:
                new_name = "fail-" + self.cf.DIR_OUTPUT.name
            self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / new_name)

    def _create_checkpoint(self, epoch, eval_result, fail=False):
        by_what = f"{self.ct.select_model_by}_valid"
        better_result: bool = eval_result is not None \
                              and eval_result[by_what] > self.best_eval_result[by_what]
        save_checkpoint: bool = fail or better_result
        checkpoint_dir = self.cf.DIR_CHECKPOINT_FAIL if fail else self.cf.DIR_CHECKPOINT
        checkpoint_file_path = self.cf.out_checkpoint_fail if fail else self.cf.out_checkpoint

        if fail:
            show_info = f"Experiment exit with Exception, checkpoint is saved at {checkpoint_file_path}"
        else:
            show_info = f'\nEpoch: {epoch} \n' + "\n".join([f' {key}: {value:.4f} ' for key, value in eval_result.items()
                                                           if isinstance(value, (int, float))])
        self.logger.info(show_info)
        if better_result:
            self.logger.info(f"\nEpoch {epoch}: {by_what} improved from {self.best_eval_result[by_what]} to {eval_result[by_what]}")
            self.logger.info("***** ***** ***** *****  save model to disk.  ***** ***** ***** *****")
            self.best_eval_result = eval_result
            with open(self.cf.out_best_eval_metrics, "w") as f:
                jsonable_best_eval_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                              for k,v in self.best_eval_result.items()}
                json.dump(jsonable_best_eval_metrics, f, indent=4)
        if save_checkpoint:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint = {
                "model_state_dict": model_to_save.state_dict(),
                "epoch": epoch,
                "best_of_which_metrics": by_what,
                "best_eval_result": dict(self.best_eval_result)}
            torch.save(checkpoint, checkpoint_file_path)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        # 变得较多的
        parser.add_argument("--debug", action='store_true',
                            help="Output dir name starts with 'debug-'.")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_pred", action='store_true',
                            help="Whether to run eval on the test set.")
        parser.add_argument("--analysis", action='store_true',
                            help="Whether to analysis model.")

        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--load_checkpoint_dir', type=str, default="",
                            help="Whether to use checkpoints to load model. If not given checkpoints, implement a new model")
        parser.add_argument('--load_checkpoint_file', type=str, default="",
                            help="Whether to use checkpoints to load model. If not given checkpoints, implement a new model")
        parser.add_argument('--commit', type=str, default='',
                            help="Current experiment's commit")
        args = parser.parse_args()
        return args

    def _get_device(self) -> torch.device:
        device = torch.device("cuda") if torch.cuda.is_available() and not self.args.no_cuda else torch.device("cpu")
        self.args.n_gpu = torch.cuda.device_count()
        return device

    def _get_data_iter(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        transform = transforms.Compose([
            # transforms.Resize((150, 200), Image.ANTIALIAS),
            transforms.Resize((227, 227), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CatDogDataset(root="./data/train/", transform=transform)
        used_data_num = self.ct.train_data_num + self.ct.valid_data_num + self.ct.test_data_num
        data_train, data_valid, data_test, data_unused = random_split(dataset, [self.ct.train_data_num, self.ct.valid_data_num, self.ct.test_data_num, len(dataset)-used_data_num])

        iter_train = DataLoader(data_train, batch_size=self.ct.train_batch_size, shuffle=True)
        iter_train_eval = DataLoader(data_train, batch_size=self.ct.eval_batch_size, shuffle=False)
        iter_valid = DataLoader(data_valid, batch_size=self.ct.eval_batch_size, shuffle=False)
        iter_test = DataLoader(data_test, batch_size=self.ct.eval_batch_size, shuffle=False)

        return iter_train, iter_train_eval, iter_valid, iter_test

    def _get_model(self, load_checkpoint_path: str or Path = None) -> nn.Module:
        # model: nn.Module = CNNModel()
        Model = MODELS[self.ct.model_name]
        model: nn.Module = Model(**self.cm.to_jsonable_dict())
        if load_checkpoint_path is not None:
            checkpoint = torch.load(load_checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    def _set_seed(self):
        """
        设置所有的随机种子
        :return:C
        """
        seed = self.ct.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _metrics(self, y_true, y_pred) -> dict:
        p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        confusion_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
        return {
            "p": p,
            "r": r,
            "f1": f1,
            "acc": acc,
            "confusion_matrix": confusion_mtx,
        }

    def _error_subset_and_index(self, dataset: Dataset, y_true, y_pred) -> Tuple[Subset, np.ndarray]:  # 预测错误的子集、index
        assert len(dataset)==len(y_true)==len(y_pred), (len(dataset), len(y_true), len(y_pred))
        error_index = np.arange(len(dataset))[np.array(y_true) != np.array(y_pred)]
        subset = Subset(dataset, error_index)
        return subset, error_index

    ##########################################################
    ## 各种tensorboard画图函数
    ##########################################################

    def _draw_each_step(self,global_step, loss, lr):
        self.writer.add_scalars(self.cf.tbx_step_train_loss, {"train loss": loss}, global_step)
        self.writer.add_scalars(self.cf.tbx_step_learning_rate, {"learning rate": lr}, global_step)

    def _draw_each_epoch(self, epoch, eval_result, model):
        # 训练曲线
        self.writer.add_scalars(self.cf.tbx_epoch_loss, {"epoch_loss_train": eval_result["loss_mean_train"],
                                                         "epoch_loss_valid": eval_result["loss_mean_valid"]}, epoch)
        self.writer.add_scalars(self.cf.tbx_epoch_acc, {"epoch_acc_train": eval_result["acc_train"],
                                                        "epoch_acc_valid": eval_result["acc_valid"]}, epoch)
        self.writer.add_scalars(self.cf.tbx_epoch_f1, {"epoch_f1_train": eval_result["f1_train"],
                                                       "epoch_f1_valid": eval_result["f1_valid"]}, epoch)
        # 混淆矩阵
        self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_train"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_train, global_step=epoch)
        self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_valid"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_valid, global_step=epoch)

        # 参数分布变化
        self._draw_parameter_distribution(model, global_step=epoch, is_train=True)

    def _draw_best(self):
        # 混淆矩阵
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_train"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_train)
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_valid"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_valid)

        # 最好的模型的网络图
        self._draw_model_graph(model=self.model, iter_data=self.iter_train)

    def _draw_model_graph(self, model: nn.Module, iter_data):  # 画模型结构图
        model.eval()
        batch = next(iter(iter_data))
        data, label, name = batch
        with torch.no_grad():
            self.writer.add_graph(model, (data, ))

    def _draw_confusion_matrix(self, confusion_matrix: np.array, graph_name: str, global_step: int=None):  # 画混淆矩阵
        categories_list = ["dog", "cat"]
        fig_confusion = fig_confusion_matrix(confusion=confusion_matrix, categories_list=categories_list)
        if global_step is None:
            self.writer.add_figure(graph_name, fig_confusion)
        else:
            self.writer.add_figure(graph_name, fig_confusion, global_step=global_step)

    def _draw_examples(self, examples: np.array, graph_name: str, global_step: int=None):  # 展示样本。可以用来展示错误样本
        fig = fig_images((torchvision.utils.make_grid(examples)))
        if global_step is None:
            self.writer.add_figure(graph_name, fig)
        else:
            self.writer.add_figure(graph_name, fig, global_step=global_step)

    def _draw_hyper_param(self, hparam_dict=None, metric_dict=None):  # 显示超参数表
        show_metric_dict = {f"hparam/{k}" : v for k,v in metric_dict.items()} if metric_dict is not None else None  # add_hparams的metric_dict的key需要与add_scalar的曲线名称不同，否则展示时可能有bug
        self.writer.add_hparams(hparam_dict=hparam_dict, metric_dict=show_metric_dict)

    def _draw_parameter_distribution(self, model: nn.Module, global_step: int=None, is_train: bool=False, is_test: bool=False, show_grad: bool=False):
        # is_train: 训练时用的选项，画出每个参数直方图随epoch的变化。  is_test：test时使用，画每个参数的直方图，以及一个把所有参数直方图按层的顺序画在一起的图  show_grad：画训练时每个参数的梯度直方图
        # 注：is_train有且仅有一个可以为true，show_grad只有在is_train情况下才起作用
        assert is_train ^ is_test, (is_train, is_test)  # ^是异或
        assert not show_grad or is_train, (show_grad, is_train)  # show_grad -> is_train
        model_name = model.__class__.__name__
        for i, (name, param) in enumerate(model.named_parameters()):
            # train和test不能画在一张图上，否则会错乱
            if show_grad and not param.requires_grad:
                continue
            show_param = param.grad.cpu().detach().numpy() if show_grad else param.cpu().detach().numpy()
            assert show_param is not None
            if is_train:
                tag = f"{model_name}/{'grad' if show_grad else 'train'}/{name}"
                self.writer.add_histogram(tag=tag, values=show_param, global_step=global_step)
            if is_test:
                self.writer.add_histogram(tag=f"{model_name}/test/{name}", values=show_param)
                self.writer.add_histogram(tag=f"{model_name}/test/all_in_one.{name.split('.')[-1]}", values=show_param, global_step=i//2)



if __name__ == '__main__':
    analysis = False
    experiment = Experiment()
    if experiment.args.do_train:
        experiment.do_train(iter_train=experiment.iter_train,
                            iter_train_eval=experiment.iter_train_eval,
                            iter_valid=experiment.iter_valid)
        experiment.analysis(iter_data=experiment.iter_test)
        analysis = True
    if experiment.args.analysis and not analysis:
        experiment.analysis(iter_data=experiment.iter_test)
        analysis = True
    if experiment.args.do_pred:
        experiment.do_predict(iter_data=experiment.iter_test)
    experiment.success()
