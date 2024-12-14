import os
import torch
import pickle
import argparse
import torchvision

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from da_baselines.data import ImagesDataset
from da_baselines.architectures import get_vit
from da_baselines.architectures import get_resnet


parser = argparse.ArgumentParser(
    description='Train baseline on Office-Like benchmarks')
parser.add_argument('--root',
                    type=str,
                    help='Root for dataset')
parser.add_argument('--benchmark',
                    type=str,
                    help='Name of benchmark')
parser.add_argument('--backbone',
                    type=str,
                    help='Type of backbone (resnet or vit)')
parser.add_argument('--size',
                    type=int,
                    help='Number of layers')
parser.add_argument('--vit_type',
                    type=str,
                    help='Type of ViT')
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--learning_rate',
                    type=float,
                    default=5e-5)
parser.add_argument('--n_iter',
                    type=int,
                    default=15)
parser.add_argument('--src',
                    type=str,
                    default='webcam',
                    help="Source domain")
parser.add_argument('--tgt',
                    type=str,
                    default='amazon',
                    help="Target domain")
parser.add_argument('--extract_features',
                    type=str,
                    default='True',
                    help="Whether or not extract features")
parser.add_argument('--out_path',
                    type=str,
                    default='./features',
                    help="path to save features")
args = parser.parse_args()

n_iter = args.n_iter
batch_size = args.batch_size
learning_rate = args.learning_rate

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.75, 1)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

train_dataset = ImagesDataset(
    root=args.root,
    dataset_name=args.benchmark,
    domains=[args.src,],
    transform=train_transforms,
    train=True,
    test=True,
    multi_source=False
)
n_classes = len(train_dataset.name2cat)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImagesDataset(
    root=args.root,
    dataset_name=args.benchmark,
    domains=[args.tgt,],
    transform=test_transforms,
    train=True,
    test=True,
    multi_source=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


if args.backbone.lower() == 'vit':
    model = get_vit(
        n_layers=args.size,
        name=args.vit_type,
        n_classes=n_classes,
        return_T=False).to('cuda')
elif args.backbone.lower() == 'resnet':
    model = get_resnet(
        resnet_size=args.size,
        n_classes=n_classes,
        return_T=False
    ).to('cuda')
else:
    raise ValueError("Expected backbone to be either vit or "
                     f"resnet, but got {args.backbone.lower()}")

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=5e-4)

criterion = torch.nn.CrossEntropyLoss()

for it in range(n_iter):
    it_loss = 0
    model.train()
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        yhat = model(x.to('cuda'))
        loss = criterion(yhat, target=y.to('cuda'))
        loss.backward()
        optimizer.step()
        it_loss += loss.item() / len(train_loader)
    print(f"Iteration {it} complete. Loss: {it_loss}")

    model.eval()
    with torch.no_grad():
        predictions = []
        ground_truth = []
        for x, y in tqdm(test_loader):
            predictions.append(model(x.to('cuda')).cpu())
            ground_truth.append(y)
        predictions = torch.cat(predictions, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        ts_loss = criterion(predictions, target=ground_truth)
        ts_acc = accuracy_score(predictions.argmax(dim=1), ground_truth)
    print("Evaluation\n")
    print(f"Loss: {ts_loss}, Accuracy: {ts_acc}\n")

if args.extract_features.lower() == 'true':
    print("Extracting features")
    dataset = {
        "Source": {"Train": None, "Test": None},
        "Target": {"Train": None, "Test": None}
    }
    model.eval()
    for domain, kind in [(args.src, 'Source'), (args.tgt, "Target")]:
        train_dataset = ImagesDataset(
            root=args.root,
            dataset_name=args.benchmark,
            domains=[domain,],
            transform=test_transforms,
            train=True,
            test=False,
            multi_source=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            Z, Y = [], []
            for x, y in train_loader:
                Z.append(model.encode(x.to('cuda')).cpu())
                Y.append(y)
            dataset[kind]["Train"] = (torch.cat(Z, dim=0),
                                      torch.cat(Y, dim=0))

        test_dataset = ImagesDataset(
            root=args.root,
            dataset_name=args.benchmark,
            domains=[domain,],
            transform=test_transforms,
            train=True,
            test=False,
            multi_source=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            Z, Y = [], []
            for x, y in test_loader:
                Z.append(model.encode(x.to('cuda')).cpu())
                Y.append(y)
            dataset[kind]["Test"] = (torch.cat(Z, dim=0),
                                     torch.cat(Y, dim=0))

    with open(
            os.path.join(args.out_path, f"{args.benchmark}_{args.src}_"
                         f"{args.tgt}_{args.backbone}_{args.size}.pkl"),
            "wb") as f:
        f.write(pickle.dumps(dataset))
