import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from callbacks import ImageLogger, SetupCallback, PeriodicCheckpoint, SaveCheckpointEveryNSteps, CodebookUsageLogger, CodebookTSNELogger

import timm
from timm.layers.pos_embed import resample_abs_pos_embed
sys.path.append('../')

from taming.data.utils import custom_collate


class DDPStrategyNoLoadOptim(DDPStrategy):
    @property
    def lightning_restore_optimizer(self):
        return False


def get_jobid():
    try:
        jobid = os.environ["PBS_JOBID"].replace(".lmbtorque.informatik.uni-freiburg.de", "")
    except KeyError:
        try:
            jobid = "DLC" + os.environ["SLURM_JOB_ID"]
        except KeyError:
            try:
                jobid = os.uname()[1]
            except KeyError:
                jobid = "local"
    return jobid


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-g",
        "--n_gpus",
        type=int,
        default=None,
        help="number of gpus per node",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=None,
        help="number of nodes",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation every n epochs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        nargs="?",
        default=True,
    )
    parser.add_argument(
        "--tsne_epoch_frequency", 
        type=int, 
        default=10,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Custom directory to save checkpoints"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="/work/dlclarge2/mutakeks-storage_titok/dino_64_new",  # default value
        help="Directory to save vqgan logs"
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        sampler = DistributedSampler(self.datasets["train"]) # if self.trainer.use_ddp else None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate, pin_memory=True, drop_last = True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, pin_memory=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    

    opt, unknown = parser.parse_known_args()

    if opt.n_gpus is None:
        # use slurm env vars if available otherwise use all CUDA_AVAILABLE gpus
        opt.n_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    if opt.n_nodes is None:
        # use slurm env vars if available otherwise set to 1
        opt.n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            logdir = os.path.dirname(opt.resume)
            ckpt = opt.resume
            print(f"Resuming from checkpoint {opt.resume}, logdir: {logdir}")
        else:
            # determine checkpoint from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            print(f"Resuming from folder {opt.resume}, logdir: {logdir}")
        opt.resume_from_checkpoint = ckpt

        # determine configs from logdir
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base

        # determine name from logdir
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        
        if opt.logdir:
            logdir = opt.logdir  # Use the custom log directory path provided by user
        else:
            # Default behavior
            name = ""
            nowname = now + name + opt.postfix + "_" + get_jobid()
            logdir = os.path.join(os.environ.get("VQ_WORK_DIR", "vqgan_logs"), nowname)
        
    print(">>> Logging to {}".format(logdir))

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        
        if opt.n_gpus == 0:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            # gpuinfo = trainer_config["gpus"]
            # print(f"Running on GPUs {gpuinfo}")
            cpu = False
        # trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # logger
        trainer_kwargs["logger"] = TensorBoardLogger(logdir, name="tb")
        print("Logging to {}".format(logdir))
        print("Logging with name {}".format("tb"))
        trainer_kwargs["strategy"] =  DDPStrategy(static_graph=True) #DDPStrategy()   #DDPStrategy(find_unused_parameters=True, static_graph=True) #"ddp"  #'ddp_find_unused_parameters_true'  DeepSpeedStrategy() # find_unused_parameters=True, static_graph=True
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = opt.n_gpus
        trainer_kwargs["num_nodes"] = opt.n_nodes
        trainer_kwargs["check_val_every_n_epoch"] = opt.check_val_every_n_epoch
        trainer_kwargs["max_epochs"] = opt.num_epochs

        # profiler        
        if opt.profiler:
            profiler = PyTorchProfiler(
            on_trace_ready = torch.profiler.tensorboard_trace_handler(logdir),
            schedule=torch.profiler.schedule(skip_first=5 ,wait=1, warmup=1, active=3, repeat=2) 
            )
            trainer_kwargs["profiler"] = profiler
        
        # adding callbacks
        trainer_kwargs["callbacks"] = [
            SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
            ImageLogger(batch_frequency=750, max_images=4, clamp=True, increase_log_steps=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=ckptdir, filename='checkpoint-{epoch}', save_top_k=3, save_on_train_epoch_end = True, save_last = True, monitor='val/rec_loss'), #monitor='val/rec_loss'
            CodebookUsageLogger(),
            PeriodicCheckpoint(every=5, end=300, dirpath=ckptdir)
        ]
        if opt.tsne_epoch_frequency is not None and opt.tsne_epoch_frequency > 0:
            trainer_kwargs["callbacks"].append(CodebookTSNELogger(epoch_frequency=opt.tsne_epoch_frequency))

        trainer = Trainer( **trainer_kwargs) #, resume_from_checkpoint=opt.resume_from_checkpoint)
        
        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        grad_acc_steps = max(config.model.params.get("grad_acc_steps", 1), config.model.get("grad_acc_steps", 1))
        #grad_acc_steps = config.model.get("grad_acc_steps", 1)
        if not cpu:
            ngpu = opt.n_gpus #len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1

        model.num_iters_per_epoch = len(data.datasets["train"]) // (config.data.params.batch_size * opt.n_gpus * grad_acc_steps)
        print("Num iters per epoch: ", model.num_iters_per_epoch)

        print("grad_acc_steps: ", grad_acc_steps)
        bs_acc_factor = max(grad_acc_steps//8, 1)
        model.learning_rate = base_lr * bs * ngpu * grad_acc_steps
        print("Setting learning rate to {:.2e} = {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr) * {} (grad accumulation factor)".format(
            model.learning_rate, ngpu, bs, base_lr, grad_acc_steps))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                if opt.resume:
                    trainer.fit(model, data, ckpt_path = opt.resume_from_checkpoint)
                else:
                    trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)