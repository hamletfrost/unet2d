import torch
import torch.nn as nn

def dice_coefficient(y_pred, y_true, epsilon=1e-6):
    y_pred = y_pred.flatten(start_dim=1)
    y_true = y_true.flatten(start_dim=1)
        
    intersection = (2 * y_pred * y_true)
    cardinality = (y_pred + y_true + epsilon)
        
    result = intersection.sum(dim=1) / cardinality.sum(dim=1)

    return result

def dice_coefficient_with_logits(y, y_true, epsilon=1e-6):
    y_pred = torch.sigmoid(y)
    
    return dice_coefficient(y_pred, y_true)

class DiceCoefficient(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficient, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        dice = dice_coefficient(y_pred, y_true, self.epsilon)
        
        return dice.mean()
        
class DiceCoefficientWithLogits(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceCoefficientWithLogits, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y, y_true):
        dice = dice_coefficient_with_logits(y, y_true, self.epsilon)
        
        return dice.mean()
