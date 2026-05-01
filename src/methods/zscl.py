from .cl_methods import ContinualLearningMethod
from ..data import RefImageData, RefImageDataloader, RefTextDataloader, RefTextData

class ZSCL(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
    
    def initialize(self, task_id):
        self.ref_image_data = RefImageData(self.wrapper.processor)
        self.ref_text_data = RefTextData(self.wrapper.processor)
        self.ref_image_dataloader = RefImageDataloader(
            self.ref_image_data,
            batch_size = self.config.datasets.batch_size,
            num_workers= self.config.datasets.num_workers,
            pin_memory= bool(self.config.datasets.pin_memory)
        )
        self.ref_text_dataloader = RefTextDataloader(
            self.ref_text_data,
            batch_size = self.config.datasets.batch_size,
            num_workers= self.config.datasets.num_workers,
            pin_memory= bool(self.config.datasets.pin_memory)
        )
        self.image_iter = iter(self.ref_image_dataloader)
        self.text_iter = iter(self.ref_text_dataloader)
    
    def compute_loss(self, images, labels, text_tokenized):
        #@ take the reference data first
        try:
            ref_images = next(self.image_iter)
        except:
            self.image_iter = iter(self.ref_image_dataloader)
            ref_images = next(self.image_iter)

        try:
            ref_texts = next(self.text_iter)
        except:
            self.text_iter = iter(self.ref_text_dataloader)
            ref_texts = next(self.text_iter)


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