from .cl_methods import ContinualLearningMethod
from ..data import get_ref_data

class ZSCL(ContinualLearningMethod):
    def __init__(self):
        super().__init__()
        self.ref_data = get_ref_data()
    
    def compute_loss(self, images, labels, text_tokenized):
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