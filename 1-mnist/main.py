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
from tensorboardX import SummaryWriter
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


from model import FNNModel, CNNModel
import config
from utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images

MODELS = {
    "CNNModel": CNNModel,
    "FNNModel": FNNModel,
}

def _wrap_handle_exception(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            return result
        except KeyboardInterrupt as e:
            self._handle_fail(keyboard_interrupt=True)
            raise
        except Exception as e:
            self._handle_fail(keyboard_interrupt=False)  # 例外处理流程
            raise
    return wrapped_func


class Experiment(object):
    def __init__(self):
        self.delete_log = True
        self.args = self._parse_args()
        self.cm = config.ConfigModel()
        self.ct = config.ConfigTrain()
        self.cf = config.ConfigFiles(commit=self.args.commit, load_checkpoint_dir=self.args.load_checkpoint_dir)

        self._set_seed()  # 设定随机种子必须在初始化model等所有步骤之前
        self.device: torch.device = self._get_device()

        self.data_train, self.data_valid, self.data_test, \
        self.iter_train, self.iter_train_eval, self.iter_valid, self.iter_test = self._get_data_iter()

        self.writer = SummaryWriter(self.cf.DIR_OUTPUT)  # 这一步自动创建了DIR_OUTPUT
        self.logger = init_logger(log_file=self.cf.out_log)
        self.best_eval_result = defaultdict(lambda: -1)

        self.model: torch.nn.Module = self._get_model(load_checkpoint_path=self.cf.load_checkpoint)
        self.model.to(self.device)
        # self.optimizer: optim.Optimizer = optim.SGD(self.model.parameters(), lr=self.ct.learning_rate, momentum=self.ct.momentum)
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters())
        # lr_lambda = lambda epoch: 1 / (1 + self.ct.lr_decay * epoch)
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

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
    def do_evaluation(self, iter_data):
        # Eval!
        self.model.eval()
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        self.logger.info("***** Running evaluation/predict *****")
        label_true, label_pred = [], []  # 真实的tag id序列，预测的tag id序列，单词原文序列
        loss_total = 0.
        data_num = 0
        for bid, batch in enumerate(tqdm(iter_data, desc="Eval")):
            with torch.no_grad():
                data, label = [t.to(self.device) for t in batch]
                pred_score = self.model(data)
                loss = F.cross_entropy(pred_score, label, reduction="sum")
                self.delete_log = False
            data_num += batch[0].shape[0]
            loss_total += loss.item()
            label_true += label.tolist()
            label_pred += torch.argmax(pred_score, dim=1).tolist()
            # tags_raw_pred += [list(self.cm.vocab_tags.keys())[tag_id] for tag_id in chain.from_iterable(decode)]
        metrics = self.metrics(y_true=label_true, y_pred=label_pred)
        error_subset, error_index = self.error_subset_and_index(dataset=iter_data.dataset, y_true=label_true,
                                                                y_pred=label_pred)
        result = {
            "loss_total": loss_total,
            "loss_mean": loss_total / data_num,
            "p": metrics["p"],
            "r": metrics["r"],
            "f1": metrics["f1"],
            "acc": metrics["acc"],
            "num_error": len(error_index),
            "confusion_matrix": metrics["confusion_matrix"],
        }
        return result, error_subset, error_index

    @_wrap_handle_exception
    def do_predict(self, iter_data):
        pass

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
        for epoch in tqdm(range(int(self.args.num_train_epochs)), desc="Train epoch"):
            for bid, batch in enumerate(tqdm(iter_train, desc="Train batch:")):
                data, label = [t.to(self.device) for t in batch]
                pred_score = self.model(data)
                loss = F.cross_entropy(pred_score, label, reduction='mean')
                # loss = loss.mean() if self.args.n_gpu > 1 else loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ct.max_grad_norm)  # 梯度剪裁，把过大的梯度限定到固定值
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self._draw_each_step(global_step=global_step, loss=loss.item(), lr=lr) # 画图
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
            self._draw_each_epoch(epoch=epoch, eval_result=eval_result)# 画图
            self._create_checkpoint(epoch=epoch, eval_result=eval_result)
        if load_best_after_train:
            if self.args.num_train_epochs>0:
                self.model = self._get_model(load_checkpoint_path=self.cf.out_checkpoint)
            self.logger.info(f"**********************  Load best model of epoch [{self.best_eval_result['epoch']}]  **********************")
        self._draw_best()
        self.writer.export_scalars_to_json(self.cf.out_scalars)
        self.cf.out_success_train.open("w").write("Train Success!!")
        self.logger.info("*****  Train Success !!!  *****")

    def success(self):
        self.writer.close()
        self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / ("success-" + self.cf.DIR_OUTPUT.name))
        self.logger.info("*****  Experiment Success  *****")

    def _handle_fail(self, keyboard_interrupt=False):
        self.writer.close()
        if self.delete_log and os.path.isdir(self.cf.DIR_OUTPUT):
            shutil.rmtree(self.cf.DIR_OUTPUT)
            return
        if self.model.training:
            self._create_checkpoint(epoch=-1, eval_result=None, fail=True)
        if not keyboard_interrupt:
            self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / ("fail-" + self.cf.DIR_OUTPUT.name))

    def _draw_each_step(self,global_step, loss, lr):
        self.writer.add_scalars(self.cf.tbx_step_train_loss, {"train loss": loss}, global_step)
        self.writer.add_scalars(self.cf.tbx_step_learning_rate, {"learning rate": lr}, global_step)

    def _draw_each_epoch(self, epoch, eval_result):
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



    def _draw_best(self):
        # 混淆矩阵
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_train"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_train)
        self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_valid"],
                                    graph_name=self.cf.tbx_best_confusion_matrix_valid)

        # 最好的模型的网络图
        self._draw_model_graph(iter_data=self.iter_train)

    def _draw_model_graph(self, iter_data):
        self.model.eval()
        batch = next(iter(iter_data))
        data, label = batch
        with torch.no_grad():
            self.writer.add_graph(self.model, (data, ))

    def _draw_confusion_matrix(self, confusion_matrix: np.array, graph_name: str, global_step: int=None):
        categories_list = list(range(10))
        fig_confusion = fig_confusion_matrix(confusion=confusion_matrix, categories_list=categories_list)
        if global_step is None:
            self.writer.add_figure(graph_name, fig_confusion)
        else:
            self.writer.add_figure(graph_name, fig_confusion, global_step=global_step)

    def _draw_examples(self, examples: np.array, graph_name: str, global_step: int=None):
        fig = fig_images((torchvision.utils.make_grid(examples)))
        if global_step is None:
            self.writer.add_figure(graph_name, fig)
        else:
            self.writer.add_figure(graph_name, fig, global_step=global_step)

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


    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser()
        # 变得较多的
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_pred", action='store_true',
                            help="Whether to run eval on the test set.")

        parser.add_argument("--num_train_epochs", default=30, type=int,
                            help="Train epochs number.")
        parser.add_argument("--train_batch_size", default=6000, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--eval_batch_size", default=12000, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument('--train_data_num', type=int, default=None,
                            help="Use a small number to test the full code")
        parser.add_argument('--eval_data_num', type=int, default=None,
                            help="Use a small number to test the full code")

        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--load_checkpoint_dir', type=str, default="",
                            help="Whether to use checkpoints to load model. If not given checkpoints, implement a new model")
        parser.add_argument('--commit', type=str, default='',
                            help="Current experiment's commit")
        args = parser.parse_args()
        return args

    def _get_device(self):
        device = torch.device("cuda") if torch.cuda.is_available() and not self.args.no_cuda else torch.device("cpu")
        self.args.n_gpu = torch.cuda.device_count()
        return device

    def _get_data_iter(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data_train_and_valid = mnist.MNIST('./data', train=True, download=True, transform=transform)
        data_test = mnist.MNIST('./data', train=False, download=True, transform=transform)

        valid_size = int(self.ct.validation_split * len(data_train_and_valid))
        train_size = len(data_train_and_valid) - valid_size
        data_train, data_valid = torch.utils.data.random_split(data_train_and_valid, [train_size, valid_size])

        iter_train = DataLoader(data_train, batch_size=self.args.train_batch_size, shuffle=False, sampler=RandomSampler(data_train))
        iter_train_eval = DataLoader(data_train, batch_size=self.args.eval_batch_size, shuffle=False, sampler=SequentialSampler(data_train))
        iter_valid = DataLoader(data_valid, batch_size=self.args.eval_batch_size, shuffle=False, sampler=SequentialSampler(data_valid))
        iter_test = DataLoader(data_test, batch_size=self.args.eval_batch_size, shuffle=False, sampler=SequentialSampler(data_test))

        return data_train, data_valid, data_test, iter_train, iter_train_eval, iter_valid, iter_test

    def _get_model(self, load_checkpoint_path: str or Path = None):
        # model: nn.Module = CNNModel()
        Model = MODELS[self.ct.model_name]
        model: nn.Module = Model()
        if load_checkpoint_path is not None:
            checkpoint = torch.load(load_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
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

    def metrics(self, y_true, y_pred):
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

    def error_subset_and_index(self, dataset: Dataset, y_true, y_pred):  # 预测错误的子集、index
        assert len(dataset)==len(y_true)==len(y_pred)
        error_index = np.arange(len(dataset))[np.array(y_true) != np.array(y_pred)]
        subset = Subset(dataset, error_index)
        return subset, error_index

    def analysis(self, iter_data):
        # 测试集上的效果
        eval_result, error_subset, error_index = self.do_evaluation(iter_data=iter_data)
        show_info = f'\nTest result: \n' + "\n".join([f' {key}: {value:.4f} ' for key, value in eval_result.items()
                                                      if isinstance(value, (int, float))])
        self.logger.info(show_info)

        # 展示错误样本
        error_loader = DataLoader(dataset=error_subset, batch_size=64)
        error_batch = iter(error_loader).next()
        images, labels = error_batch
        self._draw_examples(images, self.cf.tbx_error_examples_test)
        pass




if __name__ == '__main__':
    experiment = Experiment()
    if experiment.args.do_train:
        experiment.do_train(iter_train=experiment.iter_train,
                            iter_train_eval=experiment.iter_train_eval,
                            iter_valid=experiment.iter_valid)
        experiment.analysis(iter_data=experiment.iter_test)

    if experiment.args.do_pred:
        experiment.do_predict(iter_data=experiment.iter_test)
    experiment.success()
