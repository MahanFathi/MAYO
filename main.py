import torch
from byol_pytorch.mayo_pytorch import MAYO
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import transforms as T
from matplotlib import pyplot as plt


resnet = models.resnet18(pretrained=True)

learner = MAYO(
    resnet,
    image_size = 128,
    hidden_layer = -3,
    eta=100,
    moving_average_decay=0.99,
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

stl10 = torchvision.datasets.STL10(
    ".",
    download=True,
    transform = T.Compose([T.ToTensor()]),
)
train_dataloader = DataLoader(stl10, batch_size=128, shuffle=True)

learner.calc_norm_values(stl10.data)


for _ in range(1000):
    for images, _ in train_dataloader:
        loss = learner(images)
        _, hidden, repr = learner.online_encoder(images)
        opt.zero_grad()
        loss.backward()
        print(loss)
        opt.step()
        # learner.update_moving_average() # update moving average of target encoder


# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')
