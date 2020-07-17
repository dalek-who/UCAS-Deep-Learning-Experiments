"""
实验过程的config
"""
#%%
from pathlib import Path
import socket
from datetime import datetime
import os\

from utils import is_jsonable

# 所有Config的基类
class Config(object):
    # 把Config转换成字典，以供保存。Config可以嵌套Config
    def to_jsonable_dict(self) -> dict:
        jsonable_dict = dict()
        config_dict = dict(**vars(self.__class__), **vars(self))
        for k, v in config_dict.items():
            if not k.startswith("__"):  # 以__开头的是python-buildin属性
                if isinstance(v, Config):
                    jsonable_dict[k] = v.to_jsonable_dict()
                elif is_jsonable(v):
                    jsonable_dict[k] = v
                elif isinstance(v, Path):
                    jsonable_dict[k] = str(v.absolute())
                else:
                    jsonable_dict[k] = str(v)
        return jsonable_dict


class ConfigTrain(Config):
    num_train_epochs: int = 3
    train_batch_size: int = 100
    eval_batch_size: int = 100
    learning_rate: float = 3e-3
    eta_min: float = 1e-4
    # max_grad_norm: float = 1.0
    # lr_decay: float = 0.05
    # momentum: float = 0.9
    # weight_decay: float = 0.1
    # adam_epsilon: float = 1e-8
    # warmup_proportion: float = 0.1
    seed: int = 42
    select_model_by: str = "acc"
    model_name: str = "TextCNN"
    optimizer_name: str = "Adam"
    lr_scheduler_name: str = "CosineAnnealingLR"
    teacher_forcing_ratio: float = 0.5
    cos_scheduler_half_period_epoch: int = 4  # cos lr scheduler从最高点到最低点用几个epoch（cos周期的一半）

    word_embedding_dim: int = 50
    word_min_count: int = 5
    train_percent: float = 1.
    valid_percent: float = 1.
    test_percent: float = 1.
    # truncate_len: int = 120
    pad_token: str = "</s>"
    unk_token: str = "<unk>"
    categories_list: list = ["bad", "good"]


class ConfigModel_TextCNN(Config):
    emb_dim: int = 50
    filter_sizes: list = [2, 3, 4]
    num_each_filter: int = 100
    dropout: float = 0.5
    num_labels: int = 2


class ConfigFiles(Config):
    def __init__(self, commit: str="", load_checkpoint_file: str="", debug: bool=False):
        super(self.__class__, self).__init__()
        self.DIR_BASE = Path(".")
        assert "main.py" in os.listdir(self.DIR_BASE), self.DIR_BASE.absolute()
        self.DIR_DATA = self.DIR_BASE / "data"
        # self.DIR_W2V_CACHE: Path = self.DIR_BASE / 'vector_cache'
        self.DIR_OUTPUT = self.DIR_BASE / f"output/{'DEBUG-' if debug else ''}{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}__{commit}"
        self.DIR_CHECKPOINT = self.DIR_OUTPUT / "checkpoint"
        self.DIR_CHECKPOINT_FAIL = self.DIR_OUTPUT / "X_checkpoint_fail"
        self.DIR_TENSORBOARD: Path = self.DIR_OUTPUT / "tbX"
        self.DIR_IMG: Path = self.DIR_OUTPUT / "img"
        self.DIR_OUTPUT_BACKUP_CODE = self.DIR_OUTPUT / "code"  # 输出的备份代码文件

        # inputs:
        #   model checkpoint
        if load_checkpoint_file:
            self.load_checkpoint: Path = Path(load_checkpoint_file)
            assert self.load_checkpoint.exists()
        else:
            self.load_checkpoint: Path = None
        #   data
        self.in_train_path: Path = self.DIR_DATA / "train.txt"
        self.in_valid_path: Path = self.DIR_DATA / "validation.txt"
        self.in_test_path: Path = self.DIR_DATA / "test.txt"
        self.in_w2v_path: Path = self.DIR_DATA / "wiki_word2vec_50.bin"

        # 不做备份的文件/目录列表：
        self.no_backup_list = ["dev", "output", "data", "dataset_old", "nohup.out", "poem.tsv", "__pycache__"]

        # outputs:
        self.out_checkpoint: Path = self.DIR_CHECKPOINT / "checkpoint.pth"
        self.out_checkpoint_fail: Path = self.DIR_CHECKPOINT_FAIL / "checkpoint.pth"
        self.out_predict_result: Path = self.DIR_OUTPUT / "predict_result.txt"
        self.out_log: Path = self.DIR_OUTPUT / "log.txt"
        self.out_args: Path = self.DIR_OUTPUT / "args.json"
        self.out_config_files: Path = self.DIR_OUTPUT / "config_files.json"
        self.out_config_model: Path = self.DIR_OUTPUT / "config_model.json"
        self.out_config_train: Path = self.DIR_OUTPUT / "config_train.json"
        self.out_best_eval_metrics: Path = self.DIR_OUTPUT / "best_eval_metrics.json"
        self.out_scalars: Path = self.DIR_OUTPUT / "all_scalars.json"
        self.out_error_examples: Path = self.DIR_OUTPUT / "error_examples.json"
        self.out_success_train: Path = self.DIR_OUTPUT / "zzz_SUCCESS_train.txt"
        self.out_success_predict: Path = self.DIR_OUTPUT / "zzz_SUCCESS_predict.txt"

        # experiment log images
        self.img_find_lr: Path = self.DIR_IMG / "find_lr.png"
        self.img_loss_and_lr_together: Path = self.DIR_IMG / "loss_and_lr_together.png"

        # tensorboardX records:
        # scalars
        self.tbx_step_train_loss: str = "loss/step_train_loss"
        self.tbx_epoch_loss_mean: str = "loss/epoch_loss_mean"
        self.tbx_epoch_loss_total: str = "loss/epoch_loss_total"
        self.tbx_loss_and_lr: str = "loss/step_loss_and_lr"  # 一个把loss和lr画在一起的图，便于比较lr和loss变化速度的关系
        self.tbx_step_learning_rate: str = "lr/step_learning_rate"
        self.tbx_epoch_acc: str = "metrics/epoch_acc"
        self.tbx_epoch_p: str = "metrics/epoch_p"
        self.tbx_epoch_r: str = "metrics/epoch_r"
        self.tbx_epoch_f1: str = "metrics/epoch_f1"
        self.tbx_scalar_grad_mean: str = "grad/mean"  # 梯度平均值
        self.tbx_scalar_grad_std: str = "grad/std"  # 梯度标准差
        self.tbx_scalar_param_mean: str = "param/mean"  # 参数平均值
        self.tbx_scalar_param_mean_std: str = "param/std"  # 参数标准差
        # image
        self.tbx_epoch_confusion_matrix_train: str = "images/confusion_matrix/epoch_train"
        self.tbx_epoch_confusion_matrix_valid: str = "images/confusion_matrix/epoch_valid"
        self.tbx_best_confusion_matrix_train: str = "images/confusion_matrix/best_train"
        self.tbx_best_confusion_matrix_valid: str = "images/confusion_matrix/best_valid"
        self.tbx_confusion_matrix_test: str = "images/confusion_matrix/test"
        self.tbx_find_lr: str = "images/find_lr"
        self.tbx_img_lr_and_loss_together: str = "images/loss_and_lr_together"
        # text
        self.tbx_train_predict: str = "text/predict_examples/train"
        self.tbx_valid_predict: str = "text/predict_examples/valid"
        self.tbx_test_predict: str = "text/predict_examples/test"
        self.tbx_predict_predict: str = "text/predict_examples/predict"
        # embedding
        self.tbx_emb: str = "emb/word_embedding"
        # distribution
        self.tbx_distribution_train_param: str = "distribution/train_param"
        self.tbx_distribution_test_param: str = "distribution/test_param"
        self.tbx_distribution_grad: str = "distribution/grad"


if __name__=="__main__":
    ct = ConfigTrain()
    cm = ConfigModel_AttentionSeq2SeqMultiLayer()
    cf = ConfigFiles()
    cf.to_jsonable_dict()
