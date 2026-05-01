from .cl_methods import ContinualLearningMethod
from copy import deepcopy
from ..data import RefImageData, RefImageDataloader, RefTextDataloader, RefTextData
import torch
from torch import nn
import torch.nn.functional as F

class ZSCL(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
        self.old_LoRAs = []

    def before_task(self, task_id):
        #@ get the reference data
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

        #@store old LoRAs
        if len(self.old_LoRAs) == 0:
            self.old_LoRA = None
            return

        old_LoRA = self.wrapper.split_and_get_lora()
        self.old_LoRA = deepcopy(old_LoRA)
        self.wrapper.load_lora(old_LoRA)
        self.old_LoRAs.append(self.old_LoRA)

    def after_task(self, task_id):
        raise NotImplementedError

    
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

        current_LoRA = self.wrapper.split_and_get_lora()
        self.wrapper.load_lora(self.old_LoRA)
        with torch.inference_mode():
            ref_images_old = self.wrapper.encode_image(ref_images)
            ref_text_old = self.wrapper.encode_text(ref_texts)
        
        self.wrapper.split_and_get_lora()
        self.wrapper.load_lora(current_LoRA)
        ref_images_new = self.wrapper.encode_image(ref_images)
        ref_text_new = self.wrapper.encode_text(ref_texts)

        temp = getattr(self.config.train, "distill_temp", 1.0)
        logit_scale = self.wrapper.model.logit_scale.exp()

        old_logits = logit_scale * ref_images_old @ ref_text_old.T
        new_logits = logit_scale * ref_images_new @ ref_text_new.T

        old_i2t = F.softmax(old_logits / temp, dim=1)
        new_i2t = F.log_softmax(new_logits / temp, dim=1)

        old_t2i = F.softmax(old_logits.T / temp, dim=1)
        new_t2i = F.log_softmax(new_logits.T / temp, dim=1)

        loss_lwf_img = -(old_i2t * new_i2t).sum(dim=1).mean()
        loss_lwf_text = -(old_t2i * new_t2i).sum(dim=1).mean()


        #@ loss_ce
        logits = self.wrapper.forward_logits(text_tokenized, images)
        loss_ce = self.criterion(logits, labels)

        #@ final loss
        loss = loss_ce + self.config.train.lambda_distill * (loss_lwf_img + loss_lwf_text)
        print(
            f"{loss_lwf_img=}\n"
            f"{loss_lwf_text=}\n"
        )
        return loss
    
