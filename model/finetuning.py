import torch

from model.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Fineturning.')
    add_management_args(parser)
    add_experiment_args(parser)
    # add_rehearsal_args(parser)
    return parser


class Finetuning(ContinualModel):
    NAME = 'finetuning'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Finetuning, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
