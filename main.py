import torch
from byol_pytorch.mayo_pytorch import MAYO
from torchvision import models

resnet = models.resnet18(pretrained=False)

learner = MAYO(
    resnet,
    image_size = 128,
    hidden_layer = -3
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

learner.calc_norm_values(torch.randn(200, 3, 96, 96))

def sample_unlabelled_images():
    return torch.randn(20, 3, 96, 96)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    _, hidden, repr = learner.online_encoder(images)
    print(hidden.shape)
    print(repr.shape)
    print("=--------=")
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')
