import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
import torch.nn.functional as F

class CLASSWeightModule(object):

    def __init__(self, update_steps: int, data_loader: DataLoader,
                 classifier: nn.Module, num_classes: int,
                 device: torch.device, temperature: Optional[float] = 0.1,):
        self.update_steps = update_steps
        self.data_loader = data_loader
        self.classifier = classifier
        self.device = device
        self.class_weight_module = ClassWeightModule(temperature)
        self.class_weight = torch.ones(num_classes).to(device)
        self.num_steps = 0

    def step(self):
        self.num_steps += 1
        if self.num_steps % self.update_steps != 0:
            all_outputs = collect_classification_results(self.data_loader, self.classifier, self.device)
            self.class_weight = self.class_weight_module(all_outputs)

    def get_class_weight_for_cross_entropy_loss(self):

        return self.class_weight

    def get_class_weight_for_adversarial_loss(self, source_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        class_weight_adv_source = self.class_weight[source_labels]
        class_weight_adv_target = torch.ones_like(class_weight_adv_source) * class_weight_adv_source.mean()
        return class_weight_adv_source,class_weight_adv_target


class ClassWeightModule(nn.Module):

    def __init__(self, temperature: Optional[float] = 0.1):
        super(ClassWeightModule, self).__init__()
        self.temperature = temperature

    def forward(self, outputs: torch.Tensor):
        outputs.detach_()
        softmax_outputs = F.softmax(outputs / self.temperature, dim=1)
        class_weight = torch.mean(softmax_outputs, dim=0)
        class_weight = class_weight / torch.max(class_weight)
        class_weight = class_weight.view(-1)
        return class_weight


def collect_classification_results(data_loader: DataLoader, classifier: nn.Module,
                                   device: torch.device) -> torch.Tensor:
    training = classifier.training
    classifier.eval()
    all_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.to(device)
            output = classifier(images)
            all_outputs.append(output)
    classifier.train(training)
    return torch.cat(all_outputs, dim=0)