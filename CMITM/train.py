import datetime
import os
from argparse import ArgumentParser

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from datasets.data_module import DataModule
from datasets.pretrain_dataset import (MultimodalPretrainingDataset, multimodal_collate_fn)
from datasets.transforms import DataTransforms
from backbones.encoder_cmae import BertEncoder, ImageEncoder
from torch import distributed as dist
import wandb
import utils_mae.lr_decay as lrd

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

os.environ['WANDB_START_METHOD'] = 'thread'

class CMITM(LightningModule):

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 144,
                 num_workers: int = 8,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim, pretrained_pth=self.hparams.pretrained_pth)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q, loss_mae, pred_mae, mask_mae, pred_feat = self.img_encoder_q(batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(pred_feat.mean(dim=1))
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_full = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q_full)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        ########### image-text contrastive loss ################
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_c = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return loss_c, acc1, acc5

    def training_step(self, batch, batch_idx):
        loss_c, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = loss_c
        
        log = {
            "train_loss": loss,
            "train_loss_c": loss_c,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
    
        loss_c, acc1, acc5 = self(
            batch, batch_idx, "valid")
        loss = loss_c

        log = {
            "val_loss": loss,
            "val_loss_c": loss_c,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        
        parameters = lrd.param_groups_lrd_moco(self, self.hparams.weight_decay, no_weight_decay_list=[],
                                               lr_layer_wise="2e-5,2e-5,2e-5")
        debugc = 1
        optimizer = torch.optim.AdamW(
            parameters,
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pretrained_pth", type=str, default="/home/local/PARTNERS/cu210/data_folder/cchen/PretrainedModel/MITM.pth")
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int, default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)

        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.0)
        parser.add_argument("--gpus", type=int, default=2)
        parser.add_argument("--strategy", type=str, default="ddp")
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = trainer.num_devices
        
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    
    parser = ArgumentParser()
    
    parser = CMITM.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = CMITM(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/CMITM/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"./results")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="CMITM", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
