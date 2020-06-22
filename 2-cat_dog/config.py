"""
实验过程的config
"""
#%%
from pathlib import Path
import socket
from datetime import datetime
import os\

from utils import is_jsonable


class Config(object):
    def to_jsonable_dict(self):
        return  {k: v if is_jsonable(v) else str(v) for k, v in vars(self.__class__).items() if not k.startswith("__")}

class ConfigTrain(Config):
    num_train_epochs: int = 10
    train_batch_size: int = 200
    eval_batch_size: int = 200
    learning_rate: float = 1e-4
    # lr_decay: float = 0.05
    # max_grad_norm: float = 5.0
    # momentum: float = 0.9
    # weight_decay: float = 0.1
    # adam_epsilon: float = 1e-8
    # warmup_proportion: float = 0.1
    seed: int = 42
    select_model_by: str = "acc"
    train_data_num: int = 2000
    valid_data_num: int = 500
    test_data_num: int = 500
    model_name: str = "AlexNet"
    optimizer_name: str = "Adam"
    # lr_scheduler_name: str="LambdaLR"



class ConfigModel(Config):
    # embedding_dim: int = 300
    # lstm_hidden_size: int = 200
    # lstm_layers: int = 2
    # dropout: float = 0.5
    # dropout_embed: float = 0.5
    # tags_num: int = 8
    # vocab_tags: dict = {'O': 0, 'I-PER': 1, 'I-ORG': 2, 'I-LOC': 3, 'I-MISC': 4, 'B-MISC': 5, 'B-ORG': 6, 'B-LOC': 7}
    tags_num: int = 2

class ConfigModel_VGG(Config):
    use_batch_norm: bool = True
    fixed_cnn_param: bool = True
    tags_num: int = 2


class ConfigModel_AlexNet(Config):
    fixed_cnn_param: bool = True
    tags_num: int = 2


class ConfigFiles(Config):
    def __init__(self, commit: str="", load_checkpoint_dir: str="", load_checkpoint_file: str="", debug: bool=False):
        assert not (load_checkpoint_dir and load_checkpoint_file), (load_checkpoint_dir, load_checkpoint_file)  # 两个至多只有一个可以非空。不允许两个都有值
        self.DIR_BASE = Path(".")
        self.DIR_DATA = self.DIR_BASE / "data"
        self.DIR_LOAD_CHECKPOINT: Path = self.DIR_BASE / load_checkpoint_dir if load_checkpoint_dir.strip() else None
        self.DIR_W2V_CACHE: Path = self.DIR_BASE / 'vector_cache'
        self.DIR_OUTPUT = self.DIR_BASE / f"output/{'DEBUG-' if debug else ''}{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}__{commit}"
        self.DIR_CHECKPOINT = self.DIR_OUTPUT / "checkpoint"
        self.DIR_CHECKPOINT_FAIL = self.DIR_OUTPUT / "X_checkpoint_fail"
        self.DIR_TENSORBOARD: Path = self.DIR_OUTPUT / "tbX"

        # inputs:
        #   model checkpoint
        if load_checkpoint_file:
            self.load_checkpoint: Path = Path(load_checkpoint_file)
            assert self.load_checkpoint.exists()
        elif self.DIR_LOAD_CHECKPOINT:
            self.load_checkpoint: Path = self.DIR_LOAD_CHECKPOINT / "checkpoint.pth"
            assert self.load_checkpoint.exists()
        else:
            self.load_checkpoint: Path = None
        #   data
        self.data_train: Path = self.DIR_DATA / 'eng.train'
        self.data_valid: Path = self.DIR_DATA / 'eng.testa'
        self.data_test: Path = self.DIR_DATA / 'eng.testb'
        self.data_pred: Path = self.DIR_DATA / 'eng.testb'

        # outputs:
        self.out_checkpoint: Path = self.DIR_CHECKPOINT / "checkpoint.pth"
        self.out_checkpoint_fail: Path = self.DIR_CHECKPOINT_FAIL / "checkpoint.pth"
        self.out_predict_result: Path = self.DIR_OUTPUT / "predict_tag.txt"
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

        # tensorboardX records:
        self.tbx_step_train_loss: str = "scalars/step_train_loss"
        self.tbx_step_learning_rate: str = "scalars/step_learning_rate"
        self.tbx_epoch_loss: str = "scalars/epoch_loss"
        self.tbx_epoch_acc: str = "scalars/epoch_acc"
        self.tbx_epoch_f1: str = "scalars/epoch_f1"
        self.tbx_epoch_confusion_matrix_train: str = "images/confusion_matrix/epoch_train"
        self.tbx_epoch_confusion_matrix_valid: str = "images/confusion_matrix/epoch_valid"
        self.tbx_best_confusion_matrix_train: str = "images/confusion_matrix/best_train"
        self.tbx_best_confusion_matrix_valid: str = "images/confusion_matrix/best_valid"
        self.tbx_confusion_matrix_test: str = "images/confusion_matrix/test"
        self.tbx_error_examples_test: str = "images/error_examples/test"



