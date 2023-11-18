from abc import ABC
from typing import Optional
import torch


class AbstractReluLayer(AbstractLayer):
    def _init_(self, layer: torch.nn.modules.ReLU,
                prev: Optional[AbstractLayer] = None, next: Optional[AbstractLayer] = None) -> None:
        super()._init_(prev = prev, next = next)
        pass

    
    def backsubstitution(self, abstractBox: AbstractBox):
        abstractBox.upperBoundBias = abstractBox.upperBoundBias + abstractBox.upperBoundWeights @ self.upperBoundBias
        abstractBox.lowerBoundBias = abstractBox.lowerBoundBias + abstractBox.lowerBoundWeights @ self.lowerBoundBias
        abstractBox.upperBoundWeights = abstractBox.upperBoundWeights @ self.upperBoundWeights
        abstractBox.lowerBoundWeights = abstractBox.lowerBoundWeights @ self.lowerBoundWeights
        return self.prev.backsubstitution(abstractBox)
        
    def forward(self):
        #Initialize the lower and upper bounds on the weights
        self.lowerBoundWeights = torch.zeros_like(self.prev.lowerBound)
        self.upperBoundWeights = torch.zeros_like(self.prev.upperBound)
        self.upperBoundBias = torch.zeros_like(self.prev.upperBound)
        self.lowerBoundBias = torch.zeros_like(self.prev.lowerBound)
        
        #Create Masks for the crossing and strictly positive intervals (don't need for negative due to 0 initialization of bounds):
        crossing = torch.logical_and((self.prev.lowerBound < torch.tensor(0.)) , (self.prev.upperBound > torch.tensor(0.)))
        positive = self.prev.lowerBound > torch.tensor(0.)

        #we have a one in the multiplication all the bounds are positive (no change to the bias):
        self.lowerBoundWeights = torch.where(positive, torch.tensor([1.]), self.lowerBoundWeights)
        self.upperBoundWeights = torch.where(positive, torch.tensor([1.]), self.upperBoundWeights)

        #calculate the slope accordingly and add the slope to the
        slope = torch.where(crossing, torch.div(self.prev.upperBound, self.prev.upperBound-self.prev.lowerBound), torch.tensor([0.]))
        self.upperBoundWeights = torch.where(crossing, slope, self.upperBoundWeights)
        #TODO: implement alpha logic
        #self.lowerBound = torch.where(crossing, "SOME ALPHA EXPRESSION", self.lowerBound)
        self.upperBoundBias = torch.where(crossing, - slope * self.prev.lowerBound, self.upperBoundBias)

        #Convert the upperBoundWeights and  lowerBoundWeights into a diagonal tensor
        self.lowerBoundWeights = torch.diag(self.lowerBoundWeights.squeeze())
        self.upperBoundWeights = torch.diag(self.upperBoundWeights.squeeze())
        self.upperBoundBias.squeeze_()
        self.lowerBoundBias.squeeze_()

        size = self.lowerBoundWeights.size(0)

        abstractBox = AbstractBox(lowerBoundWeights = torch.eye(size),
                                  upperBoundWeights = torch.eye(size),                                 
                                  lowerBoundBias = torch.zeros(size),
                                  upperBoundBias = torch.zeros(size))
        self.lowerBound, self.upperBound = self.backsubstitution(abstractBox)
        assert torch.sum(self.upperBound < self.lowerBound) == 0, "The lower bound exceeds the upperbound"
        return self.next.forward()