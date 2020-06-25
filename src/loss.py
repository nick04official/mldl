# Custom Losses used for the auxiliary motion segmentation task

import torch, torch.nn

class MSClassificationLoss(torch.nn.Module):

    def __init__(self, alpha=1):
        """Loss for the RGB Network when using MMAPS

        Args:
            alpha (int, optional): Weight of the loss of the auxiliary task. Defaults to 1.
        """
        super(MSClassificationLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):
        # Main classification loss for each sample
        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        # Mean loss across all pixels of all frames for the auxiliary task
        ce = torch.nn.CrossEntropyLoss(reduction='mean')

        # Compute the loss for each sample -> I get an array where each
        # element represents the main classification loss,
        # i.e. the loss due to predicting a certain label for a
        # certain sample
        losses = ce_classification(pred_label, y_label)


        # I just reshape pred_mmap (predicted motion maps) and y_mmap (g.thruth) so that
        # y_mmap -> (batch, pixel)
        # pred_mmap -> (batch, pixel, scores)
        pred_mmap = pred_mmap.permute(1, 0, 2)
        pred_mmap = pred_mmap.reshape(pred_mmap.size(0), pred_mmap.size(1)*49, 2)
        y_mmap = y_mmap.view(y_mmap.size(0), -1)

        # For each sample, I compute the mean cross entropy loss for the predicted
        # motion maps, and I sum it to the classification loss
        for i in range(pred_mmap.size(0)):
            losses[i] += self.alpha * ce(pred_mmap[i], y_mmap[i].long())

        return losses

class MSRegressionClassificationLoss(torch.nn.Module):
    
    def __init__(self, alpha=1):
        """Regression Loss for the RGB Network when using MMAPS

        Args:
            alpha (int, optional): Weight of the loss of the auxiliary task. Defaults to 1.
        """
        super(MSRegressionClassificationLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):

        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        regression_loss = torch.nn.MSELoss(reduction='mean')

        losses = ce_classification(pred_label, y_label)

        pred_mmap = pred_mmap.permute(1, 0, 2)
        pred_mmap = pred_mmap.reshape(pred_mmap.size(0), pred_mmap.size(1)*49, 2)

        # Activation function: I restrain all the values between 0 and 1
        pred_mmap = (torch.tanh(pred_mmap.sum(dim=-1)) + 1) / 2
        pred_mmap = pred_mmap.view(pred_mmap.size(0), pred_mmap.size(1)*49)

        y_mmap = y_mmap.view(y_mmap.size(0), -1)

        for i in range(pred_mmap.size(0)):
            losses[i] += self.alpha * regression_loss(pred_mmap[i], y_mmap[i].float())

        return losses

class ClassificationLoss(torch.nn.Module):
    
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):
        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        losses = ce_classification(pred_label, y_label)

        return losses