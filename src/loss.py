import torch, torch.nn

class MSClassificationLoss(torch.nn.Module):
    
    def __init__(self):
        super(MSClassificationLoss, self).__init__()
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):
        # Loss di ciascun sample
        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        # Loss media per ciascun pixel di tutti i frame di un sample
        ce = torch.nn.CrossEntropyLoss(reduction='mean')

        # Calcolo la loss di ciascun sample -> ottengo vettore dove ogni
        # elemento è la loss dovuta alla predizione dell'azione di
        # quel sample
        losses = ce_classification(pred_label, y_label)

        # Risistemo pred_mmap e y_mmap in modo da avere (batch, pixel) per le label
        # e (batch, pixel, outputs) per gli output
        # Per ciascun sample calcolo la loss media di tutti i pixel per tutti i frame
        pred_mmap = pred_mmap.permute(1, 0, 2)
        #pred_mmap = pred_mmap.reshape(pred_mmap.size(0), 49*7, 2)
        pred_mmap = pred_mmap.reshape(pred_mmap.size(0), pred_mmap.size(1)*49, 2)
        y_mmap = y_mmap.view(y_mmap.size(0), -1)

        for i in range(pred_mmap.size(0)):
            losses[i] += ce(pred_mmap[i], y_mmap[i].long())

        return losses


class MSRegressionClassificationLoss(torch.nn.Module):
    
    def __init__(self):
        super(MSRegressionClassificationLoss, self).__init__()
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):
        # Loss di ciascun sample
        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        # Loss media per ciascun pixel di tutti i frame di un sample
        regression_loss = torch.nn.MSELoss(reduction='mean')

        # Calcolo la loss di ciascun sample -> ottengo vettore dove ogni
        # elemento è la loss dovuta alla predizione dell'azione di
        # quel sample
        losses = ce_classification(pred_label, y_label)

        # Risistemo pred_mmap e y_mmap in modo da avere (batch, pixel) per le label
        # e (batch, pixel, outputs) per i target
        # Per ciascun sample calcolo la loss media di tutti i pixel
        pred_mmap = pred_mmap.permute(1, 0, 2)
        pred_mmap = pred_mmap.reshape(pred_mmap.size(0), 49*7, 2)
        # Restringo i valori tra 0 e 1
        pred_mmap = (torch.tanh(pred_mmap.sum(dim=-1)) + 1) / 2
        pred_mmap = pred_mmap.view(pred_mmap.size(0), 49*7)

        y_mmap = y_mmap.view(y_mmap.size(0), -1)

        for i in range(pred_mmap.size(0)):
            losses[i] += regression_loss(pred_mmap[i], y_mmap[i].float())

        return losses

class ClassificationLoss(torch.nn.Module):
    
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        
    def forward(self, pred_label, y_label, pred_mmap, y_mmap):
        # Loss di ciascun sample
        ce_classification = torch.nn.CrossEntropyLoss(reduction='none')

        losses = ce_classification(pred_label, y_label)

        return losses