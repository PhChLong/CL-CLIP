from .cl_methods import ContinualLearningMethod
from ..data import get_ref_data, TaskData, TaskDataLoader

class ZSCL(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
    
    def initialize(self, task_id):
        self.ref_data = TaskData(get_ref_data())
        self.ref_dataloader = TaskDataLoader(
            self.ref_data,
            batch_size = self.config.datasets.batch_size,
            num_workers= self.config.datasets.num_workers,
            pin_memory= bool(self.config.datasets.pin_memory)
        )
        self.iter = iter(self.ref_dataloader)
    
    def compute_loss(self, images, labels, text_tokenized):
        try:
            images, labels = next(self.iter)
        except:
            self.iter = iter(self.ref_dataloader)
            images, labels = next(self.iter)


        #@loss_lw_img
        loss_lwf_img = 0

        #@loss_lw_img
        loss_lwf_text = 0

        #@ loss_ce
        logits = self.wrapper.forward_logits(text_tokenized, images)
        loss_ce = self.criterion(logits, labels)

        #@ final loss
        loss = loss_ce + self.config.train.lambda_distill * (loss_lwf_img + loss_lwf_text)
        return loss