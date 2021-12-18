import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



def get_backbone(args):
    if args.backbone == "resnet18":
        backbone = torchvision.models.resnet18(pretrained=bool(args.backbone_pretrained),
                                               progress=False).to(args.device)
    elif args.backbone == "resnet50":
        backbone = torchvision.models.resnet50(pretrained=bool(args.backbone_pretrained),
                                               progress=False).to(args.device)
    else:
        raise Exception(f"Backbone architecture {args.backbone} not supported!")
    return backbone


class RegressionNet(nn.Module):
    def __init__(self, learner, args, representation_count=512, num_classes=10):
        super().__init__()
        self.model = args.model
        self.learner = learner
        self.fc = nn.Linear(representation_count, num_classes)

    def forward(self, x):
        if self.model == "supervised":
            x = self.learner(x)
        elif self.model in ["byol", "simsiam"]:
            with torch.no_grad():
                _, embed = self.learner(x, return_embedding=True)
                embed = embed.squeeze().squeeze()
            x = self.fc(embed)
        elif self.model in ["mayo", "random"]:
            with torch.no_grad():
                embed = self.learner(x, return_embedding=True)
                embed = embed.squeeze().squeeze()
            x = self.fc(embed)
        else:
            raise Exception(f"Model {self.model} not supported!")
        return x


def get_reg_optim(net, args):
    if args.reg_optim == "adam":
        optim = torch.optim.Adam(net.parameters(), lr=args.reg_optim_lr)
    else:
        raise Exception(f"Regression optimizer {args.reg_optim} not supported!")
    return optim


def get_reg_dataloaders(args):
    if args.dataset == "STL10":
        trainset = torchvision.datasets.STL10(
            root=args.dataset_path,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            split="train"
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.ssl_batch_size,
                                                  shuffle=True, num_workers=6)

        testset = torchvision.datasets.STL10(
            root=args.dataset_path,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            split="test"
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.ssl_batch_size,
                                                  shuffle=True, num_workers=6)
    else:
        Exception(f"Regression dataset {args.dataset} not supported!")
    return trainloader, testset


def test_reg_acc(epoch, net, testloader, args):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return {
        "reg_test_epoch": epoch,
        "reg_acc": (100 * correct / total)
    }


def train_regression(learner, args):
    # Get Regression Model
    net = RegressionNet(learner, args)
    net_optim = get_reg_optim(net, args)
    net_criterion = nn.CrossEntropyLoss()

    # Get Regression DataLoaders
    reg_trainloader, reg_testloader = get_reg_dataloaders(args)

    # Train Regression
    reg_metrics = []
    reg_metrics.append(test_reg_acc(0, net, reg_testloader, args))
    for epoch in range(args.reg_epochs):
        for i, data in enumerate(reg_trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            net_optim.zero_grad()
            outputs = net(inputs)
            loss = net_criterion(outputs, labels)
            loss.backward()
            net_optim.step()

        if epoch % args.save_every == 0:
            reg_metrics.append(test_reg_acc(epoch+1, net, reg_testloader, args))
            save_file(reg_metrics, "jsonl", "reg_metrics.jsonl", args)
            save_file(net, "torch", "reg_net.pth", args)

    reg_metrics.append(test_reg_acc(args.reg_epochs+1, net, reg_testloader, args))
    save_file(reg_metrics, "jsonl", "reg_metrics.jsonl", args)
    save_file(net, "torch", "reg_net.pth", args)


def get_model(backbone, args):
    if args.model == "byol":
        from byol_pytorch.byol_pytorch import BYOL
        learner = BYOL(
            backbone,
            image_size=args.image_size,
            hidden_layer=args.hidden_layer,
            moving_average_decay= args.moving_average_decay,
            use_momentum=bool(args.use_momentum)
        ).to(args.device)
    elif args.model == "simsiam":
        from byol_pytorch.byol_pytorch import BYOL
        learner = BYOL(
            backbone,
            image_size=args.image_size,
            hidden_layer=args.hidden_layer,
            use_momentum=False
        ).to(args.device)
    elif args.model == "mayo":
        from byol_pytorch.mayo_pytorch import MAYO
        learner = MAYO(
            backbone,
            image_size=args.image_size,
            hidden_layer=args.hidden_layer,
            eta=args.eta,
            moving_average_decay=args.moving_average_decay,
            use_momentum=bool(args.use_momentum)
        ).to(args.device)
    elif args.model == "random":
        from byol_pytorch.mayo_pytorch import MAYO
        learner = MAYO(
            backbone,
            image_size=args.image_size,
            hidden_layer=args.hidden_layer,
            eta=args.eta,
            moving_average_decay=args.moving_average_decay,
            use_momentum=bool(args.use_momentum)
        ).to(args.device)
    else:
        raise Exception(f"Model {args.model} not supported!")
    return learner


def get_ssl_optim(learner, args):
    if args.ssl_optim == "adam":
        optim = torch.optim.Adam(learner.parameters(), lr=args.ssl_optim_lr)
    else:
        raise Exception(f"SSL optimizer {args.ssl_optim} not supported!")
    return optim


def get_ssl_trainloader(args):
    if args.dataset == "STL10":
        trainset = torchvision.datasets.STL10(
            root=args.dataset_path,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            split=args.dataset_ssl_split
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.ssl_batch_size,
                                                  shuffle=True, num_workers=6)
    else:
        Exception(f"SSL dataset {args.dataset} not supported!")
    return trainloader


def save_file(obj, type, filename, args):
    if type == "jsonl":
        with open(os.path.join(args.output_path, filename), "w") as f:
            for d in obj:
                f.write(json.dumps(d) + "\n")
    elif type == "torch":
        torch.save(obj.state_dict(), os.path.join(args.output_path, filename))
    else:
        Exception(f"Type {type} not supported!")


def train_ssl(learner, learner_optim, ssl_trainloader, args):
    # Set metric logger
    ssl_metrics = []

    ssl_epochs = 0 if args.model == "random" else args.ssl_epochs
    for epoch in range(ssl_epochs):
        epoch_start_time = time.time()
        running_loss = 0.
        for i, data in enumerate(ssl_trainloader, 0):
            # Get inputs
            inputs, _ = data
            inputs = inputs.to(args.device)

            # Cal loss and optimize
            loss = learner(inputs)
            learner_optim.zero_grad()
            loss.backward()
            learner_optim.step()

            running_loss += loss.item()

        if (args.model != "simsiam") and (not args.use_momentum):
            learner.update_moving_average()
        epoch_end_time = time.time()
        ssl_metrics.append({
            "epoch": epoch,
            "duration": ((epoch_start_time - epoch_end_time) / 60.),
            "average_loss" : (running_loss / len(ssl_trainloader))
        })
        if epoch % args.save_every == 0:
            save_file(ssl_metrics, "jsonl", "ssl_metrics.jsonl", args)
            save_file(learner, "torch", "learner.pth", args)

    save_file(ssl_metrics, "josnl", "ssl_metrics.jsonl", args)
    save_file(learner, "torch", "learner.pth", args)
    return learner


def build_and_train(args):
    # Get backbone arch
    backbone = get_backbone(args)

    if args.model == "supervised":
        train_regression(backbone, args)
        return

    # Get SSL Model
    learner = get_model(backbone, args)
    learner_optim = get_ssl_optim(learner, args)

    # Get SSL Dataset TrainLoader
    ssl_trainloader = get_ssl_trainloader(args)

    # Train SSL
    learner = train_ssl(learner, learner_optim, ssl_trainloader, args)

    # Train Regression
    train_regression(learner, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str,
                        choices=["supervised", "byol", "simsiam", "mayo", "random"], default="mayo")
    parser.add_argument("--backbone", type=str, choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--backbone_pretrained", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=["STL10"], default="STL10")
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--dataset_ssl_split", type=str,
                        choices=["train", "test", "unlabeled", "train+unlabeled"], default="train+unlabeled")
    parser.add_argument("--eta", type=float, default=100.)
    parser.add_argument("--moving_average_decay", type=float, default=0.99)
    parser.add_argument("--use_momentum", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--ssl_batch_size", type=int, default=128)
    parser.add_argument("--reg_batch_size", type=int, default=128)
    parser.add_argument("--hidden_layer", default=-3)
    parser.add_argument("--ssl_epochs", type=int, default=400)
    parser.add_argument("--reg_epochs", type=int, default=100)
    parser.add_argument("--ssl_optim", type=str, default="adam")
    parser.add_argument("--ssl_optim_lr", type=float, default=3e-4)
    parser.add_argument("--reg_optim", type=str, default="adam")
    parser.add_argument("--reg_optim_lr", type=float, default=3e-4)
    parser.add_argument("--run_id", type=str, default="tmp")
    parser.add_argument("--path", type=str, default="./experiments")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    # Set output path
    args.output_path = os.path.join(args.path, args.run_id)
    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.output_path, "config.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

        # Set device
    if torch.cuda.is_available() and (args.device != "cpu"):
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    # Train and Evaluate the Model
    build_and_train(args)
