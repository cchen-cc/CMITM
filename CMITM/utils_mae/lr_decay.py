import json
from pdb import set_trace

import torch.distributed
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch


def param_groups_lrd_moco(model, weight_decay=0.05, no_weight_decay_list=[],
                          layer_decay=.75, lr_layer_wise="2.0e-6,2.0e-6,2.0e-5"):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if lr_layer_wise != "":
        num_layers = len(model.img_encoder_q.model.blocks)
        lr_layer_wise = [float(lr) for lr in lr_layer_wise.split(",")]
        assert num_layers % len(lr_layer_wise) == 0
        block_size_each = num_layers // len(lr_layer_wise)
        # TODO: check if this work
        layer_scales = [lr_layer_wise[0], ] + [lr_layer_wise[i // block_size_each] for i in range(num_layers)]
    else:
        num_layers = len(model.base_encoder.blocks) + 1
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        if "img_encoder_q.model." in n:
            layer_id = get_layer_id_for_vit(n.replace("img_encoder_q.model.", ""), num_layers)
            # print("name of {} is in layer {}".format(n, layer_id))
        else:
            layer_id = num_layers
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if torch.distributed.get_rank() == 0:
                print("lr scale of {} is {}".format(group_name, this_scale))

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, fc_scale=1, log=None):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if "head" in n:
            group_name += "_head"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            if "head" in group_name:
                this_scale *= fc_scale
                if log is not None:
                    print("scale of group {} is {}".format(group_name, this_scale))

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

class CosineAnnealingWarmupRestarts_dynamic(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = []
        for param_group in optimizer.param_groups:
            self.base_max_lr.append(param_group['lr_scale'])
        self.max_lr = []
        for param_group in optimizer.param_groups:
            self.max_lr.append(param_group['lr_scale'])
        
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts_dynamic, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self, group_idx):
        if self.step_in_cycle == -1:
            return self.base_lrs[group_idx]
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr[group_idx] - self.base_lrs[group_idx])*self.step_in_cycle / self.warmup_steps + self.base_lrs[group_idx]]
        else:
            return [self.base_lrs[group_idx] + (self.max_lr[group_idx] - self.base_lrs[group_idx]) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

        for max_lr_idx in range(len(self.max_lr)):
            self.max_lr[max_lr_idx] = self.base_max_lr[max_lr_idx] * (self.gamma**self.cycle)
            
        self.last_epoch = math.floor(epoch)

        for group_idx in range(len(self.optimizer.param_groups)):
            param_group = self.optimizer.param_groups[group_idx]
            lr = self.get_lr(group_idx)
            param_group['lr'] = lr