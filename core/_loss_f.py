import torch


class LossFunction:
    def __init__(self,
                 prediction,
                 target,
                 device_for_training,
                 model_output_class: int = 1,
                 semantic_binary: bool = True,
                 ):
        self.prediction = prediction
        self.device = device_for_training
        self.target = target
        self.num_classes = model_output_class
        self.semantic_binary = semantic_binary

    def forward(self):
        if self.semantic_binary:
            return self.dice_loss(self.prediction, self.target)
        return self.categorical_dice_loss()

    @staticmethod
    def dice_loss(predictions, targets, alpha=1e-5):
        if predictions.sum().item() == targets.sum().item() == 0:
            return alpha
        intersection = 2. * (predictions * targets).sum()
        denomination = (torch.square(predictions) + torch.square(targets)).sum()
        dice_loss = 1 - torch.mean((intersection + alpha) / (denomination + alpha))

        return dice_loss

    def categorical_dice_loss(self):
        pr, tr = self.prepare_for_multiclass_loss_f()
        losses = 0
        for num_category in range(self.num_classes):
            categorical_target = torch.where(tr == num_category, 1, 0)
            categorical_prediction = pr[num_category]
            losses += self.dice_loss(categorical_prediction, categorical_target).to(self.device)

        return losses

    def prepare_for_multiclass_loss_f(self):
        prediction_prepared = torch.squeeze(self.prediction, 0)
        target_prepared = torch.squeeze(self.target, 0)
        target_prepared = torch.squeeze(target_prepared, 0)
        target_prepared = torch.argmax(target_prepared, 0)

        return prediction_prepared, target_prepared
