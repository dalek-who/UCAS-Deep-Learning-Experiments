"""
实验数据记录器
记录各种画图、实验记录功能
主要是tensorboard
"""

import shutil
import os
from pathlib import Path
from logging import Logger
from collections import defaultdict
from typing import Tuple, Union, Dict, NoReturn
import json
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

try:
    from .config import ConfigTrain, ConfigFiles, ConfigModel_TextCNN
    from .utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images
except:
    from config import ConfigTrain, ConfigFiles, ConfigModel_TextCNN
    from utils import is_jsonable, init_logger, fig_confusion_matrix, fig_images


class ExperimentLogWriter(object):
    def __init__(self, ct: ConfigTrain, cf: ConfigFiles, cm, args):
        self.ct = ct
        self.cf = cf
        self.cm = cm
        self.args = args
        self.writer = SummaryWriter(self.cf.DIR_TENSORBOARD)
        self.all_scalars = defaultdict(dict)  # tensorboard画scalar时，同时记录下scalar的值，方便日后画图
        self.cf.DIR_IMG.mkdir(exist_ok=True)  # matplotlib图片输出路径

    def close(self):
        self.writer.close()

    def export_scalars_to_json(self, file) -> NoReturn:
        """
        以json形式保存所有tensorboard的scalars数据点
        tensorboardX是有此功能的，但是官方的torch.utils.tensorboard未整合此功能，所以我手动添加了一个
        :param file:
        :return:
        """
        with open(file, "w") as f:
            json.dump(self.all_scalars, f)

    def draw_each_step(self,global_step, loss, lr) -> NoReturn:
        """
        每个train step需要画的图
        :param global_step: 当前是累计第几个step
        :param loss: loss
        :param lr: 学习率（画学习率图可以避免学习率变成负值的问题）
        :return:
        """
        self.writer.add_scalars(self.cf.tbx_step_train_loss, {"train loss": loss}, global_step)
        self.all_scalars["train loss"][global_step] = loss

        self.writer.add_scalars(self.cf.tbx_step_learning_rate, {"learning rate": lr}, global_step)
        self.all_scalars["learning rate"][global_step] = lr

        self.writer.add_scalars(self.cf.tbx_loss_and_lr, {"loss": loss, "lr": lr}, global_step=global_step)

    def draw_each_epoch(self, epoch, eval_result, model) -> NoReturn:
        """
        每个train epoch需要画的图
        :param epoch: 当前是第几个Epoch
        :param eval_result: metrics评估结果字典
        :param model: 当前的模型，用来画权重、梯度分布的
        :return:
        """
        self.writer.add_scalars(self.cf.tbx_epoch_loss, {"epoch_loss_train": eval_result["loss_mean_train"],
                                                         "epoch_loss_valid": eval_result["loss_mean_valid"]}, epoch)
        self.all_scalars["epoch_loss_train"][epoch] = eval_result["loss_mean_train"]
        self.all_scalars["epoch_loss_valid"][epoch] = eval_result["loss_mean_valid"]

        self.writer.add_scalars(self.cf.tbx_epoch_acc, {"epoch_acc_train": eval_result["acc_train"],
                                                        "epoch_acc_valid": eval_result["acc_valid"]}, epoch)
        self.all_scalars["epoch_acc_train"][epoch] = eval_result["acc_train"]
        self.all_scalars["epoch_acc_valid"][epoch] = eval_result["acc_valid"]

        self.writer.add_scalars(self.cf.tbx_epoch_f1, {"epoch_f1_train": eval_result["f1_train"],
                                                       "epoch_f1_valid": eval_result["f1_valid"]}, epoch)
        self.all_scalars["epoch_f1_train"][epoch] = eval_result["f1_train"]
        self.all_scalars["epoch_f1_valid"][epoch] = eval_result["f1_valid"]

        # 序列生成任务不需要展示这项
        # 混淆矩阵
        # self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_train"],
        #                             graph_name=self.cf.tbx_best_confusion_matrix_train, global_step=epoch)
        # self._draw_confusion_matrix(confusion_matrix=eval_result["confusion_matrix_valid"],
        #                             graph_name=self.cf.tbx_best_confusion_matrix_valid, global_step=epoch)

        # 参数分布变化
        self.draw_parameter_distribution(model, global_step=epoch, is_train=True)

    def draw_best(self) -> NoReturn:
        """
        训练完后最好的模型，画这些东西
        :return:
        """
        # 序列生成任务不需要展示这项
        # 混淆矩阵
        # self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_train"],
        #                             graph_name=self.cf.tbx_best_confusion_matrix_train)
        # self._draw_confusion_matrix(confusion_matrix=self.best_eval_result["confusion_matrix_valid"],
        #                             graph_name=self.cf.tbx_best_confusion_matrix_valid)

        pass

    def draw_model_graph(self, model: nn.Module, input_to_model, device) -> NoReturn:
        """
        tensorboard画模型结构图（注：目前还不支持cuda，数据需要转换到cpu上。add_graph默认把模型放在cpu上，不用手动处理模型
        :param model:
        :param iter_data:
        :return:
        """
        model.eval()
        def draw(model, input_to_model, data_device, model_device):
            input_to_model = [t.to(data_device) for t in input_to_model]
            model.to(model_device)
            with torch.no_grad():
                self.writer.add_graph(model, input_to_model=input_to_model)

        for data_device, model_device in [(device, device), ("cpu", device), ("cpu", "cpu")]:
            try:
                draw(model, input_to_model, data_device, model_device)
                break
            except RuntimeError as e:
                print(f"draw_model_graph:\n{e}")
                if (data_device, model_device) == ("cpu", "cpu"):
                    raise e

        model.to(device)

    def draw_confusion_matrix(self, confusion_matrix: np.array, graph_name: str, global_step: int=None) -> NoReturn:
        """
        tensorboard画混淆矩阵
        注：如果混淆矩阵维数过大，会报Locator attempting to generate xxx ticks ([-1.0, ..., xxx]), which exceeds Locator.MAXTICKS，且保存会非常慢
        :param confusion_matrix: 混淆矩阵
        :param graph_name: 混淆矩阵图的名字
        :param global_step:  第几个global_step，可以查看混淆矩阵随着step的变化
        :return:
        """
        categories_list = ["dog", "cat"]
        fig_confusion = fig_confusion_matrix(confusion=confusion_matrix, categories_list=categories_list)
        if global_step is None:
            self.writer.add_figure(graph_name, fig_confusion)
        else:
            self.writer.add_figure(graph_name, fig_confusion, global_step=global_step)

    def draw_examples(self, examples: list, graph_name: str, global_step: int=None) -> NoReturn:
        """
        展示样本。正确样本、错误样本都可以
        :param examples: 样本
        :param graph_name: 图的名字
        :param global_step: 第几个global_step
        :return:
        """
        text = "  \n".join(examples[: min(10, len(examples))])  # add_text是以markdown格式显示的，所以\n前先放两个空格才能换行
        self.writer.add_text(tag=graph_name, text_string=text, global_step=global_step)


    def draw_hyper_param(self, hparam_dict=None, metric_dict=None) -> NoReturn:
        """
        在tensorboard中显示超参数表
        :param hparam_dict: 超参数
        :param metric_dict: 评测指标
        :return:
        """
        show_metric_dict = {f"hparam/{k}" : v for k,v in metric_dict.items()} if metric_dict is not None else None  # add_hparams的metric_dict的key需要与add_scalar的曲线名称不同，否则展示时可能有bug
        self.writer.add_hparams(hparam_dict=hparam_dict, metric_dict=show_metric_dict)

    def draw_parameter_distribution(self, model: nn.Module, global_step: int=None, is_train: bool=False, is_test: bool=False, show_grad: bool=False) -> NoReturn:
        """
        tensorboard画模型的参数分布
        每个分布图在tensorboard中都会自动产生两张图：histogram和distribution，一个是正视图，另一个是俯视图
        :param model: 模型
        :param global_step: 当前是第几个global_step（只在is_train=True时有用）
        :param is_train: 训练时用的选项，画出每个参数直方图随epoch的变化。【注：is_train有且仅有一个可以为true，show_grad只有在is_train情况下才起作用】
        :param is_test: test时使用，画每个参数的直方图，以及一个把所有参数直方图按层的顺序画在一起的图(all_in_one)
        :param show_grad: 画训练时每个参数的梯度直方图
        :return:
        """
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
                all_in_one_name = name.split('.')[-1].split("_")[0]  # "weight" or "bias"
                self.writer.add_histogram(tag=f"{model_name}/test/all_in_one.{all_in_one_name}", values=show_param, global_step=i//2)

    def draw_find_lr_plot(self, lrs, losses):
        """
        用来寻找适合的学习率范围的函数
        :return:
        """
        lrs, losses = np.array(lrs), np.array(losses)
        fig, ax_loss = plt.subplots()
        ax_loss.plot(np.log10(lrs)[10:-5], losses[10:-5], label="loss", color="blue")
        ax_loss.set_xlabel('log10 lr')  # x轴title
        ax_loss.set_ylabel("loss")  # y轴title
        ax_loss.legend()  # 图例
        ax_loss.set_title("find learning rate")  # y可以调整标题的位置

        # lr图
        ax_log10_loss = ax_loss.twinx()  # 叠加在原ax_loss图上，共享x轴
        ax_log10_loss.plot(np.log10(lrs)[10:-5], np.log10(losses)[10:-5], label="log10 loss", color="red")
        ax_log10_loss.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))  # y轴数字用保留两位小数的科学技术法表示
        ax_log10_loss.set_ylabel("log10 loss")  # y轴标签
        ax_log10_loss.legend()  # 添加图例

        # 保存与显示
        fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
        fig.savefig(self.cf.img_find_lr)
        self.writer.add_figure(self.cf.tbx_find_lr, fig)

        # fig.canvas.set_window_title("find learning rate")  # 窗口fig的title
        # fig.show()

    def draw_embedding(self):
        """
        利用tensorboard画Embedding
        :return:
        """
        # todo
        pass

    def draw_attention(self):
        """
        画Attention
        :return:
        """
        # todo
        pass

    def draw_loss_and_lr_together(self):
        """
        把loss和learning rate画在一张图上，便于比较适合的学习率
        :return:
        """
        global_step = [int(step) for step in self.all_scalars['train loss']]
        step_loss = list(self.all_scalars['train loss'].values())
        step_lr = list(self.all_scalars['learning rate'].values())
        epoch_loss = list(self.all_scalars['epoch_loss_train'].values())
        # 没直接记录一个epoch到底有几个step，把整个step
        step_of_each_epoch = len(global_step) // len(epoch_loss)  # 一个epoch有几个step
        epoch_to_global_step = [(i + 1) * step_of_each_epoch - 1 for i in range(len(epoch_loss))]  # 每个epoch对应的是第几个step

        # epoch和step的loss图
        fig, ax_loss_step = plt.subplots()
        ax_loss_step.plot(global_step, step_loss, label="train loss")
        ax_loss_step.plot(epoch_to_global_step, epoch_loss, label="epoch loss")
        ax_loss_step.set_xlabel("global step")  # x轴的标签
        ax_loss_step.set_ylabel("loss")  # y轴的标签
        ax_loss_step.legend()  # 添加图例
        ax_loss_step.grid()  # 背景显示网格线

        # 叠加一个新的图用来标epoch x轴刻度
        # epoch loss不要画在这个子图上，否则图例与train loss的不在一个方格里
        epoch_gap = 10  # 每隔多少epoch画个刻度
        ax_loss_epoch = ax_loss_step.twiny()  # 叠加在原ax_loss_step图上，共享y轴
        ax_loss_epoch.set_xlim(ax_loss_step.get_xlim())  # x轴对齐
        ax_loss_epoch.set_xticks(epoch_to_global_step[::epoch_gap])  # 画epoch刻度线
        ax_loss_epoch.set_xticklabels(list(range(len(epoch_loss)))[::epoch_gap])  # 标epoch数值
        ax_loss_epoch.set_xlabel("epoch")  # x轴的标签

        # lr图
        ax_lr_step = ax_loss_step.twinx()  # 叠加在原ax_loss_step图上，共享x轴
        ax_lr_step.plot(global_step, step_lr, color="red")
        ax_lr_step.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))  # y轴数字用保留两位小数的科学技术法表示
        ax_lr_step.set_ylabel("learning rate")  # y轴标签
        ax_lr_step.legend(["lr"])  # 添加图例

        # 保存与显示
        fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
        fig.savefig(self.cf.img_loss_and_lr_together)
        self.writer.add_figure(self.cf.tbx_img_lr_and_loss_together, fig)

    def back_up_code(self):
        """
        把当前代码文件保存到输出log目录里，作为备份代码
        打包成压缩包，防止误操作修改
        :return:
        """
        os.makedirs(self.cf.DIR_OUTPUT_BACKUP_CODE, exist_ok=True)
        for path in os.listdir(self.cf.DIR_BASE):
            if path not in self.cf.no_backup_list:
                if os.path.isdir(path):
                    shutil.copytree(path, self.cf.DIR_OUTPUT_BACKUP_CODE / path)
                else:
                    shutil.copy(path, self.cf.DIR_OUTPUT_BACKUP_CODE / path)
        # 将代码打压缩包，防止备份代码文件被误删或修改
        shutil.make_archive(self.cf.DIR_OUTPUT_BACKUP_CODE, "gztar", self.cf.DIR_OUTPUT_BACKUP_CODE)

    def back_up_config(self):
        """
        保存各种config
        :return:
        """
        with open(self.cf.out_config_files, "w") as f_cf, \
                open(self.cf.out_config_train, "w") as f_ct, \
                open(self.cf.out_config_model, "w") as f_cm, \
                open(self.cf.out_args, "w") as f_a:
            json.dump(vars(self.args), f_a, indent=4)
            json.dump(self.cf.to_jsonable_dict(), f_cf, indent=4)
            json.dump(self.ct.to_jsonable_dict(), f_ct, indent=4)
            json.dump(self.cm.to_jsonable_dict(), f_cm, indent=4)


