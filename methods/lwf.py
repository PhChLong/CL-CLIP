
from models import LoRAAdapter, CLIPWrapper
from methods.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from copy import deepcopy
from data import TaskData, TaskDataLoader

class TextModelWithProjection(nn.Module):
    def __init__(self, text_model, text_projection):
        super().__init__()
        self.text_model = text_model
        self.text_projection = text_projection

    def forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output      # [B, 512]
        text_embeds = self.text_projection(text_features)  # [B, 512]
        return text_embeds
class LwF_LoRA(BaseTrainer):
    def __init__(self, wrapper: CLIPWrapper, config: Config):
        super().__init__(wrapper, config)
        self.add_lora()
        self.wrapper = wrapper
        self.vision_model = wrapper.model.vision_model
        self.org_proj = wrapper.model.visual_projection
        self.text_model_include_proj = TextModelWithProjection(wrapper.model.text_model, wrapper.model.text_projection)
        # lưu head theo task
        self.task_heads = nn.ModuleDict()
        # setpoint cho old tasks:
        # {old_task_id: tensor [N, D]}
        self.old_task_setpoints = {}
        # nhiệt độ distillation
        self.distill_temp = self.config.train.distill_temp
        # trọng số loss cũ
        self.lambda_old = getattr(self.config.train, "lambda_old", 1.0)
        #id của các task đã train
        self.trained_task_id = []
    def add_lora(self):
        device = self.wrapper.model.device
        # ? Freeze model lại
        for params in self.wrapper.parameters():
            params.requires_grad = False

        #? thêm LoRA vào các layer q_proj, v_proj trong model
        for i in range(self.config.model.num_layers):
            for layer_type in ['q_proj', 'v_proj']:
                org_attention_layer = self.wrapper.model.vision_model.encoder.layers[i].self_attn
                org_layer = getattr(org_attention_layer, layer_type)
                setattr(org_attention_layer, layer_type, LoRAAdapter(org_layer, r = self.config.train.r))
                
        #? add model to GPU
        self.wrapper.model.to(device)
    
    def _build_new_head_if_needed(self, task_id):
        #? neếu chưa từng train trên task này thì sẽ tạo head mới
        if task_id not in self.trained_tasks_ids:
            new_head = deepcopy(self.wrapper.model.visual_projection)
            new_head.load_state_dict(self.wrapper.model.visual_projection.state_dict())
            new_head = new_head.to(self.wrapper.model.device)

            #? chỉ train head mới này
            #! cần check lại vì mình nhớ là train hết mọi head
            for p in new_head.parameters():
                p.requires_grad = True

            self.task_heads[str(task_id)] = new_head
    
    @torch.no_grad()
    def _init_old_tasks_setpoint(self, dataloader, device):
        if len(self.task_heads) == 0:
            self.old_task_setpoints = {}
            return

        self.old_task_setpoints = {str(tid):[] for tid in self.trained_task_id}

        self.vision_model.eval()
        for image, text in dataloader:
            image_backbone_encoded = self.vision_model(image).pooler_output
            for 



    def train(self, task, task_id = None):
        optimizer = self.optimizer(self.wrapper.model.parameters(),
                                   lr = float(self.config.train.lr),
                                   weight_decay = float(self.config.train.weight_decay))

        criterion = nn.CrossEntropyLoss()
        prompts = [f"a photo of a {name}" for name in task['label_names']]

        #* =======================Data and Dataloader============================
        train_data = TaskData(task, "train", image_processor= self.wrapper.processor.image_processor)
        test_data = TaskData(task, "test", image_processor= self.wrapper.processor.image_processor)
        train_loader = TaskDataLoader(
            train_data,
            batch_size= self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory= True)
        test_loader = TaskDataLoader(
            test_data,
            batch_size = self.config.datasets.batch_size,
            num_workers=self.config.datasets.num_workers,
            pin_memory=True
        )

        device = self.wrapper.model.device
        #* =====================Compute outputs for prev_heads==================
        self._init_old_tasks_setpoint(train_loader, device)
        
    