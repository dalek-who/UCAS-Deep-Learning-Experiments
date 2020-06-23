from typing import Tuple, Union, NoReturn, Dict, List
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, SequentialSampler, RandomSampler, Subset, Dataset
import re
import argparse
from functools import wraps
import random
import os
import numpy as np
from tqdm import tqdm
import logging
import json
from itertools import chain
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from pathlib import Path
import shutil
from torchtext.vocab import Vocab
from torchtext.data import Iterator, BucketIterator
from collections import defaultdict

try:
    from .models.AttentionSeq2Seq import AttentionSeq2Seq
    from .models.AttentionSeq2SeqMultiLayer import AttentionSeq2SeqMultiLayer
    from .models.LanguageModel import LanguageModel
    from .config import ConfigTrain, ConfigFiles, ConfigModel_AttentionSeq2Seq, ConfigModel_AttentionSeq2SeqMultiLayer, ConfigModel_LanguageModel
    from .utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images
    from .dataset import get_dataset_and_vocab, get_dataset_and_vocab_language_model
    from .ExperimentLogWriter import ExperimentLogWriter
except:
    from models.AttentionSeq2Seq import AttentionSeq2Seq
    from models.AttentionSeq2SeqMultiLayer import AttentionSeq2SeqMultiLayer
    from models.LanguageModel import LanguageModel
    from config import ConfigTrain, ConfigFiles, ConfigModel_AttentionSeq2Seq, ConfigModel_AttentionSeq2SeqMultiLayer, ConfigModel_LanguageModel
    from utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images
    from dataset import get_dataset_and_vocab, get_dataset_and_vocab_language_model
    from ExperimentLogWriter import ExperimentLogWriter


MODELS = {
    "AttentionSeq2Seq": AttentionSeq2Seq,
    "AttentionSeq2SeqMultiLayer": AttentionSeq2SeqMultiLayer,
    "LanguageModel": LanguageModel,
}

MODEL_CONFIGS = {
    "AttentionSeq2Seq": ConfigModel_AttentionSeq2Seq,
    "AttentionSeq2SeqMultiLayer": ConfigModel_AttentionSeq2SeqMultiLayer,
    "LanguageModel": ConfigModel_LanguageModel,
}

# 例外处理流程
def _wrap_handle_exception(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            return result
        except KeyboardInterrupt as e:
            self.logger.error(e)
            self._handle_fail(keyboard_interrupt=True)
            raise
        except (Exception, SystemExit) as e:
            self.logger.error(e)
            self._handle_fail(keyboard_interrupt=False)  # 例外处理流程
            raise
    return wrapped_func


class Experiment(object):
    def __init__(self):
        self.delete_log = True
        self.args = self._parse_args()
        self.DEBUG = self.args.debug
        self.ct = ConfigTrain()
        ModelConfig = MODEL_CONFIGS[self.ct.model_name]
        self.cm = ModelConfig()
        self.cf = ConfigFiles(commit=self.args.commit, load_checkpoint_file=self.args.load_checkpoint_file, debug=self.DEBUG)

        self._set_seed()  # 设定随机种子必须在初始化model等所有步骤之前
        self.device: torch.device = self._get_device()

        self.iter_train, self.iter_train_eval, self.iter_valid, self.iter_test, self.iter_predict, self.vocab_input, self.vocab_output = self._get_data_iter()

        self.writer: ExperimentLogWriter = ExperimentLogWriter(ct=self.ct, cf=self.cf, cm=self.cm, args=self.args)  # 不要和下面的代码换位。因为整个输出目录都是ExperimentLogWriter创建的
        self.writer.back_up_code()  # 备份代码
        self.writer.back_up_config()  # 备份config
        self.logger = init_logger(log_file=self.cf.out_log)
        self.best_eval_result = defaultdict(lambda: -1)

        self.model: torch.nn.Module = self._get_model(load_checkpoint_path=self.cf.load_checkpoint)
        self.model.to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=self.ct.learning_rate)
        # lr_lambda = lambda epoch: 1 / (1 + self.ct.lr_decay * epoch)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.iter_train) * self.ct.cos_scheduler_half_period_epoch, eta_min=self.ct.eta_min)
        self._check_experiment_and_config()  # 检查实例化的对象和Config声明中的是否一致

    @_wrap_handle_exception
    def do_predict(self, iter_data, has_label: bool, write_file: bool=False) -> Tuple[np.ndarray, np.ndarray, float, int, List[str]]:
        self.model.eval()
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        self.logger.info("***** Running evaluation/predict *****")
        label_true, label_pred = [], []  # 真实的tag id序列，预测的tag id序列，单词原文序列
        loss_total = 0.
        data_num = 0
        poem_predict_list = []
        for bid, batch in enumerate(tqdm(iter_data, desc="Eval")):
            with torch.no_grad():
                loss, batch_size, src, trg, trg_for_loss, pred_score, pred_score_for_loss = \
                    self._forward_batch(batch, eval=True, has_label=has_label)
                self.delete_log = False
            data_num += batch_size
            loss_total += loss.item() * batch_size
            label_true += trg_for_loss.tolist()
            label_pred += torch.argmax(pred_score_for_loss, dim=1).tolist()
            # tags_raw_pred += [list(self.cm.vocab_tags.keys())[tag_id] for tag_id in chain.from_iterable(decode)]
            trg_predict = pred_score[1:].argmax(dim=2)  # 第0个位置是全0向量，但是取argmax时可能得到任意一个index，于是将其跳过
            poem_predict_list += self._tensor_to_poem(src, trg_predict)
        label_true, label_pred = np.array(label_true), np.array(label_pred)
        # 写入预测结果文件
        if write_file:
            text = "\n".join(poem_predict_list)
            with open(self.cf.out_predict_result, "w") as f:
                f.write(text)
        return label_true, label_pred, loss_total, data_num, poem_predict_list

    @_wrap_handle_exception
    def do_evaluation(self, iter_data) -> Tuple[Dict[str, int or float], List[str]]:
        # Eval!
        label_true, label_pred, loss_total, data_num, poem_predict_list = self.do_predict(iter_data, has_label=True)   # type: np.ndarray, np.ndarray, float, int, list
        metrics = self._metrics(y_true=label_true, y_pred=label_pred)
        # error_subset, error_index = self._error_subset_and_index(dataset=iter_data.dataset, y_true=label_true, y_pred=label_pred)  # 序列生成任务不适合用这个函数找错误样本
        result: dict = {
            "loss_total": float(loss_total),
            "loss_mean": float(loss_total / data_num),
            "p": float(metrics["p"]),
            "r": float(metrics["r"]),
            "f1": float(metrics["f1"]),
            "acc": float(metrics["acc"]),
            # "num_error": int(len(error_index)),
            "num_error": int((label_true != label_pred).sum()),
            # "confusion_matrix": metrics["confusion_matrix"],  # 序列生成任务，因为词表很大，以字为单位的confusion_matrix过大
        }
        # return result, error_subset, error_index
        return result, poem_predict_list

    @_wrap_handle_exception
    def do_train(self, iter_train, iter_train_eval, iter_valid, load_best_after_train=True) -> NoReturn:
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
                loss, batch_size, _, _, _, _, _ = self._forward_batch(batch, train=True, has_label=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ct.max_grad_norm)  # 梯度剪裁，把过大的梯度限定到固定值
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.writer.draw_each_step(global_step=global_step, loss=loss.item(), lr=lr) # 画图
                if bid==0:
                    self.writer.draw_parameter_distribution(self.model, global_step=epoch, is_train=True, show_grad=True)  # 画梯度分布图。需要在zero_grad之前
                self.optimizer.step()
                self.model.zero_grad()
                self.delete_log = False
                global_step += 1
                self.lr_scheduler.step()  # Update learning rate each global step
            eval_train, poem_predict_list_train = self.do_evaluation(iter_train_eval)
            eval_valid, poem_predict_list_valid = self.do_evaluation(iter_valid)
            eval_result = dict(**{k+"_train": v for k,v in eval_train.items()},
                               **{k+"_valid": v for k,v in eval_valid.items()})
            eval_result["epoch"] = epoch
            self.writer.draw_each_epoch(epoch=epoch, eval_result=eval_result, model=self.model)# 画图
            self.writer.draw_examples(examples=poem_predict_list_train, graph_name=self.cf.tbx_train_predict, global_step=epoch)
            self.writer.draw_examples(examples=poem_predict_list_valid, graph_name=self.cf.tbx_valid_predict, global_step=epoch)
            self._create_checkpoint(epoch=epoch, global_step=global_step, eval_result=eval_result)
        if load_best_after_train:
            if num_train_epochs>0:
                self.model = self._get_model(load_checkpoint_path=self.cf.out_checkpoint)
            self.logger.info(f"**********************  Load best model of epoch [{self.best_eval_result['epoch']}]  **********************")
        # self.writer.draw_best()
        self.writer.draw_loss_and_lr_together()
        self.writer.export_scalars_to_json(self.cf.out_scalars)
        self.cf.out_success_train.open("w").write("Train Success!!")
        self.logger.info("*****  Train Success !!!  *****")

    @_wrap_handle_exception
    def analysis(self, iter_data) -> NoReturn:
        """
        测试集上的效果
        :param iter_data:
        """
        eval_result, poem_predict_list = self.do_evaluation(iter_data=iter_data)  # type: dict, list
        eval_result_info = {k:v for k,v in eval_result.items() if isinstance(v, (int, float, str))}
        show_info = f'\nTest result: \n' + "\n".join([f' {key}: {value:.4f} ' for key, value in eval_result_info.items()])
        self.logger.info(show_info)

        # 展示超参数
        hparam_dict = dict(self.ct.to_jsonable_dict(), **self.cm.to_jsonable_dict())
        self.writer.draw_hyper_param(hparam_dict=hparam_dict, metric_dict=eval_result_info)

        # 展示各层参数的分布情况
        self.writer.draw_parameter_distribution(self.model, is_test=True)

        # # 最好的模型的网络图
        # # 暂时用处不大，而且容易爆显存，先放弃了
        # batch = next(iter(iter_data))
        # src, src_len = batch.input
        # trg, trg_len = batch.output
        # input_to_model = [src, src_len, trg, torch.tensor(0)]
        # self.writer.draw_model_graph(model=self.model, input_to_model=input_to_model, device=self.device)

        # 展示预测样本
        self.writer.draw_examples(examples=poem_predict_list, graph_name=self.cf.tbx_test_predict)

        # 序列生成任务不需要分析这个
        # 展示错误样本
        # error_loader = DataLoader(dataset=error_subset, batch_size=64)
        # error_batch = iter(error_loader).next()
        # images, labels, names = error_batch
        # self.writer.draw_examples(images, self.cf.tbx_error_examples_test)

        # 序列生成任务不需要分析这个
        # 保存错误样本的名字和index，便于查阅
        # error_name_list = [example.name for example in error_subset]
        # error_index_list = error_index.tolist()
        # with open(self.cf.out_error_examples, "w") as f:
        #     json.dump({"name_list": error_name_list, "index_list": error_index_list}, f)

    def find_lr(self, iter_train, init_value=1e-8, final_value=10., beta=0.98):
        """
        找个比较适合的学习率
        算法来自https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param final_value:
        :param beta:
        :return:
        """
        num = len(iter_train) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []
        for batch_num, batch in enumerate(tqdm(iter_train, desc="Train batch:"), start=1):
            loss, batch_size, _, _, _, _, _ = self._forward_batch(batch, train=True, has_label=True)
            self.delete_log = False
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                # return log_lrs, losses
                break
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(np.log10(lr))
            # Do the SGD step
            loss.backward()
            self.optimizer.step()
            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        self.writer.draw_find_lr_plot(logs=log_lrs, losses=losses)
        # return log_lrs, losses

    def success(self) -> NoReturn:
        """
        实验成功的后续流程
        :return:
        """
        # 关闭tensorboard
        self.writer.close()
        # 保存tensorboard画的所有scalar数据点
        self.writer.export_scalars_to_json(self.cf.out_scalars)
        # 为了防止早期bug生成过多无意义的log，如果在网络forward之前报bug直接删掉log
        if self.delete_log and os.path.isdir(self.cf.DIR_OUTPUT):
            shutil.rmtree(self.cf.DIR_OUTPUT)
            return
        # log文件夹重命名，前面加上success
        if self.cf.DIR_OUTPUT.name.startswith("DEBUG-"):  # DEBUG模式
            new_name = self.cf.DIR_OUTPUT.name.replace("DEBUG-", "DEBUG-success-")
        else:  # 普通运行模式
            new_name = "success-" + self.cf.DIR_OUTPUT.name
        self.cf.DIR_OUTPUT.rename(self.cf.DIR_OUTPUT.parent / new_name)
        self.logger.info("*****  Experiment Success  *****")

    def _handle_fail(self, keyboard_interrupt=False) -> NoReturn:
        """
        实验因为Exception失败的处理流程
        :param keyboard_interrupt: 是否是被KeyboardInterrupt人工中断的
        :return:
        """
        # 关闭tensorboard
        self.writer.close()
        # 保存tensorboard画的所有scalar数据点
        self.writer.export_scalars_to_json(self.cf.out_scalars)
        # 为了防止早期bug生成过多无意义的log，如果在网络forward之前报bug直接删掉log
        if self.delete_log and os.path.isdir(self.cf.DIR_OUTPUT):
            shutil.rmtree(self.cf.DIR_OUTPUT)
            return
        # 训练时意外退出需要保存checkpoint
        if self.model.training:
            self._create_checkpoint(epoch=-1, global_step=-1, eval_result=None, fail=True)
        # 如果不是被KeyboardInterrupt人工中断的，log文件夹重命名，前面加上fail
        if not keyboard_interrupt:
            parent_dir = self.cf.DIR_OUTPUT.parent
            if self.cf.DIR_OUTPUT.name.startswith("DEBUG-"):  # DEBUG模式
                new_name = self.cf.DIR_OUTPUT.name.replace("DEBUG-", "DEBUG-fail-")
            else:  # 普通运行模式
                new_name = "fail-" + self.cf.DIR_OUTPUT.name
            self.cf.DIR_OUTPUT.rename(parent_dir / new_name)

    def _forward_batch(self, batch, train=False, eval=False, has_label=True):
        fake_trg_len = 100
        assert train ^ eval, (train, eval)  # train或eval有且只有一个可以为True
        if train:
            assert has_label
            self.model.train()
        else:
            self.model.eval()

        src, src_len = batch.input  # 创建iteration时已经放到device上了
        # src = [src len, batch size], src_len = [batch size]

        batch_size = src.shape[1]
        if has_label:
            trg, trg_len = batch.output
            # trg = [trg len, batch size]

        else:  # 为了无标签时predict生成的"伪标签"
            trg = torch.ones(fake_trg_len, batch_size, dtype=torch.int64, device=self.device) * self.vocab_output[self.ct.output_init_token]
            # trg = [trg len, batch size]
            trg_len = torch.ones(batch_size, dtype=torch.int64, device=self.device) * fake_trg_len

        teacher_forcing_ratio = self.ct.teacher_forcing_ratio if train else 0
        pred_score = self.model(src, src_len, trg, torch.tensor(teacher_forcing_ratio))
        # pred_score = [trg len, batch size, output vocab dim]

        output_vocab_dim = pred_score.shape[-1]
        pred_score_for_loss = pred_score[1:].view(-1, output_vocab_dim)
        # pred_score_for_loss = [(trg len-1) * batch size, output vocab dim]
        trg_for_loss = trg[1:].view(-1)
        # trg_for_loss = [(trg len -1) * batch size]

        # predict时，用伪标签计算的loss并无意义，但是为了流程统一，也计算个loss
        batch_mean_loss = F.cross_entropy(pred_score_for_loss, trg_for_loss, reduction='mean', ignore_index=self.vocab_output[self.ct.pad_token])
        # batch_mean_loss = batch_mean_loss.mean() if self.args.n_gpu > 1 else batch_mean_loss
        return batch_mean_loss, batch_size, src, trg, trg_for_loss, pred_score, pred_score_for_loss

    def _create_checkpoint(self, epoch, global_step, eval_result, fail=False) -> NoReturn:
        """
        将模型参数以state_dict的形式保存为checkpoint
        :param epoch: 当前是第几个Epoch
        :param global_step: 当前是第几个global step
        :param eval_result: 当前的评测结果
        :param fail: 是否遇到Exception（遇到Exception退出时也会保存checkpoint）
        :return:
        """
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
                'optimizer':
                    {"class": self.optimizer.__class__.__name__,
                     "state_dict": self.optimizer.state_dict()},
                "lr_scheduler": {},
                "epoch": epoch,
                "global_step": global_step,
                "best_of_which_metrics": by_what,
                "best_eval_result": dict(self.best_eval_result)}
            if hasattr(self, "lr_scheduler"):
                checkpoint["lr_scheduler"] = {"class": self.lr_scheduler.__class__.__name__,
                                              "state_dict": self.lr_scheduler.state_dict()},
            torch.save(checkpoint, checkpoint_file_path)

    def _parse_args(self):
        """
        定义、解析命令行参数
        :return: 命令行参数
        """
        parser = argparse.ArgumentParser()
        # 变得较多的
        parser.add_argument("--debug", action='store_true',
                            help="Output dir name starts with 'debug-'.")
        parser.add_argument("--find_lr", action='store_true',
                            help="Find a proper learning rate")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_pred", action='store_true',
                            help="Whether to run eval on the test set.")
        parser.add_argument("--analysis", action='store_true',
                            help="Whether to analysis model.")

        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--load_checkpoint_file', type=str, default="",
                            help="Whether to use checkpoints to load model. If not given checkpoints, implement a new model")
        parser.add_argument('--commit', type=str, default='',
                            help="Current experiment's commit")
        args = parser.parse_args()
        return args

    def _get_device(self) -> torch.device:
        """
        获取需要使用的device
        :return: device对象
        """
        device = torch.device("cuda") if torch.cuda.is_available() and not self.args.no_cuda else torch.device("cpu")
        self.args.n_gpu = torch.cuda.device_count()
        return device

    def _get_data_iter(self) -> Tuple[Iterator, Iterator, Iterator, Iterator, Iterator, Vocab, Vocab]:
        """
        加载数据集，划分成训练、验证、测试、预测几个集合
        按batch_size等超参数包装成数据加载器DataLoader或Iterator（torchtext提供的nlp数据加载器）
        :return:
            iter_train: 训练集迭代器
            iter_train_eval: 用于eval的训练集迭代器（batch_size和iter_train不同
            iter_valid: 验证集迭代器
            iter_test: 测试集迭代器
            iter_predict: 预测集迭代器（与iter_test相比，iter_predict可能没有label）
            vocab_input: 输入数据的词表(Vocab对象)
            vocab_output:  输出数据的词表(Vocab对象)
        """
        # 数据集预处理与加载
        if self.ct.model_name in ("AttentionSeq2Seq", "AttentionSeq2SeqMultiLayer"):
            dataset_train, dataset_valid, dataset_test, dataset_predict, vocab_input, vocab_output = get_dataset_and_vocab(
                input_path=self.cf.in_train_valid_test_data_path,
                preprocessed_data_path=self.cf.in_preprocessed_train_valid_test_data_path,
                predict_input_path=self.cf.in_predict_data_path,
                word_embedding_dim=self.ct.word_embedding_dim,
                window_size=self.ct.window_size,
                word_min_count=self.ct.word_min_count,
                train_percent=self.ct.train_percent,
                valid_percent=self.ct.valid_percent,
                test_percent=self.ct.test_percent,
                pad_token=self.ct.pad_token,
                unk_token=self.ct.unk_token,
                input_init_token=self.ct.input_init_token,
                output_init_token=self.ct.output_init_token,
                output_eos_token=self.ct.output_eos_token,
            )
        elif self.ct.model_name in ("LanguageModel", ):
            dataset_train, dataset_valid, dataset_test, dataset_predict, vocab_input, vocab_output = get_dataset_and_vocab_language_model(
                input_path=self.cf.in_train_valid_test_data_path,
                preprocessed_data_path=self.cf.in_preprocessed_train_valid_test_data_path,
                predict_input_path=self.cf.in_predict_data_path,
                word_embedding_dim=self.ct.word_embedding_dim,
                window_size=self.ct.window_size,
                word_min_count=self.ct.word_min_count,
                train_percent=self.ct.train_percent,
                valid_percent=self.ct.valid_percent,
                test_percent=self.ct.test_percent,
                pad_token=self.ct.pad_token,
                unk_token=self.ct.unk_token,
                input_init_token=self.ct.input_init_token,
                output_init_token=self.ct.output_init_token,
                output_eos_token=self.ct.output_eos_token,
            )
        else:
            raise ValueError

        # 按batch_size等超参数封装成数据加载器
        iter_train = BucketIterator(
            dataset_train, batch_size=self.ct.train_batch_size, train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.input), device=self.device)
        iter_train_eval, iter_valid, iter_test = (
            BucketIterator(
                dataset, batch_size=self.ct.eval_batch_size, train=False, sort_within_batch=True,
                sort_key=lambda x: len(x.input), device=self.device)
            for dataset in (dataset_train, dataset_valid, dataset_test)
        )
        iter_predict = Iterator(
            dataset_predict, batch_size=self.ct.eval_batch_size, train=False, shuffle=False,
            sort=False, device=self.device)
        return iter_train, iter_train_eval, iter_valid, iter_test, iter_predict, vocab_input, vocab_output

    def _get_model(self, load_checkpoint_path: str or Path = None) -> nn.Module:
        """
        实例化模型、初始化模型参数
        :param load_checkpoint_path: 若不为None，则加载load_checkpoint_path文件初始化模型参数
        :return: 实例化的、参数初始化的模型
        """
        Model = MODELS[self.ct.model_name]
        if self.ct.model_name in ("AttentionSeq2Seq", "AttentionSeq2SeqMultiLayer"):
            model_args = dict(
                enc_input_dim=len(self.vocab_input), enc_vocab=self.vocab_input,
                dec_output_dim=len(self.vocab_output), dec_vocab=self.vocab_output,
                **self.cm.to_jsonable_dict()
            )
        elif self.ct.model_name in ("LanguageModel", ):
            model_args = dict(
                vocab_size=len(self.vocab_input), vocab=self.vocab_input,
                **self.cm.to_jsonable_dict()
            )
        else:
            raise ValueError
        model: nn.Module = Model(**model_args)
        if load_checkpoint_path is not None:
            checkpoint = torch.load(load_checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    def _set_seed(self) -> NoReturn:
        """
        设置所有的随机种子
        :return:
        """
        seed = self.ct.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _check_experiment_and_config(self) -> NoReturn:
        """
        检查实验的设置和Config中的声明是否一致
        :return:
        """
        # 检查model
        assert self.model.__class__.__name__ == self.ct.model_name, \
            (self.model.__class__.__name__, self.ct.model_name)
        # 检查optimizer
        assert self.optimizer.__class__.__name__ == self.ct.optimizer_name, \
            (self.optimizer.__class__.__name__, self.ct.optimizer_name)
        # 检查lr_scheduler
        if hasattr(self, "lr_scheduler"):
            assert self.lr_scheduler.__class__.__name__ == self.ct.lr_scheduler_name, \
                (self.lr_scheduler.__class__.__name__, self.ct.lr_scheduler_name)

        # Config中的设定是否矛盾。依据具体任务而定
        if self.ct.model_name in ("AttentionSeq2Seq", "AttentionSeq2SeqMultiLayer"):
            assert self.ct.word_embedding_dim == self.cm.enc_emb_dim, \
                (self.ct.word_embedding_dim, self.cm.enc_emb_dim)
            assert self.ct.word_embedding_dim == self.cm.dec_emb_dim, \
                (self.ct.word_embedding_dim, self.cm.dec_emb_dim)
            assert self.ct.pad_token == self.cm.pad_token, \
                (self.ct.pad_token, self.cm.pad_token,)
            assert self.ct.output_init_token == self.cm.output_init_token, \
                (self.ct.output_init_token, self.cm.output_init_token)
            assert self.ct.output_eos_token == self.cm.output_eos_token, \
                (self.ct.output_eos_token == self.cm.output_eos_token)
        elif self.ct.model_name in ("LanguageModel", ):
            assert self.ct.word_embedding_dim == self.cm.embed_dim, \
                (self.ct.word_embedding_dim, self.cm.embed_dim)
        else:
            raise ValueError

    def _metrics(self, y_true, y_pred) -> dict:
        """
        评估模型预测结果（metrics）
        :param y_true: 正确标签
        :param y_pred: 预测标签
        :return:
        """
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

    def _error_subset_and_index(self, dataset: Dataset, y_true, y_pred) -> Tuple[Subset, np.ndarray]:
        """
        模型预测错误的子集、样本index
        :param dataset: 样本集
        :param y_true:  样本集上的标签
        :param y_pred:  样本集上的预测结果
        :return:
            subset: 模型预测错误的子集（Subset对象）
            error_index：错误预测的样本在dataset上的index
        """
        assert len(dataset)==len(y_true)==len(y_pred), (len(dataset), len(y_true), len(y_pred))
        error_index = np.arange(len(dataset))[np.array(y_true) != np.array(y_pred)]
        subset = Subset(dataset, error_index)
        return subset, error_index

    def _tensor_to_poem(self, src: torch.LongTensor, trg: torch.LongTensor) -> List[str]:
        # src: [src_seq_len, batch_size]
        # trg: [trg_seq_len, batch_size]
        func_src_itos = np.vectorize(lambda x: self.vocab_input.itos[x])
        func_trg_itos = np.vectorize(lambda x: self.vocab_output.itos[x])
        array_poem_src = func_src_itos(src.cpu().numpy().T)
        array_poem_trg = func_trg_itos(trg.cpu().numpy().T)
        batch_size = array_poem_trg.shape[0]
        array_output_init_token = func_trg_itos(np.ones((batch_size, 1), dtype=np.int64) * self.vocab_output[self.ct.output_init_token])
        array_split_poem = np.concatenate([array_poem_src, array_output_init_token, array_poem_trg], axis=1)
        # array_split_poem: [batch_size, src_seq_len + trg_seq_len]

        def func_poem_str(row):
            poem = "".join(filter(lambda word: word not in (self.ct.pad_token, self.ct.input_init_token), row))
            poem = poem.split(self.ct.output_eos_token)[0]  # 可能预测出多个<EOP>，只取第一个EOP前的作为答案
            return poem

        poem_list = np.apply_along_axis(func1d=func_poem_str, axis=1, arr=array_split_poem).tolist()
        return poem_list


if __name__ == '__main__':
    analysis = False
    experiment = Experiment()
    if experiment.args.find_lr:
        experiment.find_lr(iter_train=experiment.iter_train)
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
        experiment.do_predict(iter_data=experiment.iter_predict, has_label=False, write_file=True)
    experiment.success()
