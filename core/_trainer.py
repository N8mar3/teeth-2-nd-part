import torch
from tqdm import tqdm
from _loss_f import LossFunction


class TrainFunction:
    def __init__(self,
                 data_loader,
                 device_for_training,
                 model_name,
                 model_name_pretrained,
                 model,
                 optimizer,
                 scale,
                 model_output_class: int = 1,
                 learning_rate: int = 1e-2,
                 num_epochs: int = 1,
                 transfer_learning: bool = False,
                 binary_loss_f: bool = True
                 ):
        self.data_loader = data_loader
        self.device = device_for_training
        self.model_name_pretrained = model_name_pretrained
        self.semantic_binary = binary_loss_f
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.num_classes = model_output_class
        self.transfer = transfer_learning
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = model
        self.scale = scale

    def forward(self):
        print('Running with:', torch.cuda.get_device_name(self.device))
        self.model.load_state_dict(torch.load(self.model_name_pretrained)) if self.transfer else None
        optimizer = self.optimizer(self.model.parameters(),
                                   lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            self.train_loop(self.data_loader, self.model,
                            optimizer, self.scale, epoch)
            torch.save(self.model.state_dict(),
                       'models/' + self.model_name
                                 + str(epoch + 1)
                                 + '_epoch.pth') if (epoch + 1) % 40 == 0 else None

    def train_loop(self, loader, model, optimizer, scales, i):
        loop, epoch_loss = tqdm(loader), 0
        loop.set_description('Epoch %i' % (self.num_epochs - i))
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(device=self.device, dtype=torch.float), \
                            targets.to(device=self.device, dtype=torch.long)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = LossFunction(predictions, targets,
                                    model_output_class=self.num_classes,
                                    device_for_training=self.device,
                                    semantic_binary=self.semantic_binary
                                    ).forward()
            scales.scale(loss).backward()
            scales.step(optimizer)
            scales.update()
            epoch_loss += (1 - loss.item()) * 100
            loop.set_postfix(loss=loss.item())
        print('Epoch-acc', round(epoch_loss / (batch_idx + 1), 2), "%")
