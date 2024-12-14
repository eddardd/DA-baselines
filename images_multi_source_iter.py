import os
import torch
import pickle
import argparse
import torchvision

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

from da_baselines.data import all_domains
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
parser.add_argument('--eval_every',
                    type=int,
                    default=1000)
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
parser.add_argument('--optimizer_name',
                    type=str,
                    default="adam",
                    help="optimizer name (either SGD or Adam)")
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

domains = all_domains[args.benchmark]
sources = [d for d in domains if d != args.tgt]

train_dataset = ImagesDataset(
    root=args.root,
    dataset_name=args.benchmark,
    domains=sources,
    transform=train_transforms,
    train=True,
    test=True,
    multi_source=False
)
n_classes = len(train_dataset.name2cat)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)

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


if args.optimizer_name.lower() == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=5e-4)


criterion = torch.nn.CrossEntropyLoss()

pbar = tqdm(range(n_iter))
for it in pbar:
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    yhat = model(x.to('cuda'))
    loss = criterion(yhat, target=y.to('cuda'))
    loss.backward()
    optimizer.step()
    pbar.set_description(f"Iteration {it} complete. Loss: {loss.item()}")

    try:
        with open(f"./logs/{args.benchmark}/{args.tgt}_train.csv", 'a') as f:
            f.write(f"{it},{loss.item()}\n")
    except FileNotFoundError:
        print(f"Error, path './logs/{args.benchmark}/{args.tgt}"
              "_train.csv' does not exist")
        print("Continuing...")

    if (it + 1) % args.eval_every == 0:
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
        print(f"Evaluation (it {it})\n")
        print(f"Loss: {ts_loss}, Accuracy: {ts_acc}\n")

        try:
            with open(
                 f"./logs/{args.benchmark}/{args.tgt}_test.csv", 'a') as f:
                f.write(f"{it},{ts_loss},{ts_acc}\n")
        except FileNotFoundError:
            print(f"Error, path './logs/{args.benchmark}/{args.tgt}_test.csv'"
                  " does not exist")
            print("Continuing...")
        model.train()

if args.extract_features.lower() == 'true':
    print("Extracting features")
    model.eval()
    dataset = {'sources': {}, 'target': None}
    for domain in sources:
        tr_loader = torch.utils.data.DataLoader(
            ImagesDataset(
                root=args.root,
                dataset_name=args.benchmark,
                domains=[domain, ],
                transform=test_transforms,
                train=True,
                test=False,
                multi_source=False
            ),
            batch_size=batch_size,
            shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ImagesDataset(
                root=args.root,
                dataset_name=args.benchmark,
                domains=[domain, ],
                transform=test_transforms,
                train=False,
                test=True,
                multi_source=False
            ),
            batch_size=batch_size,
            shuffle=False)

        _xtr, _xts, _ytr, _yts = [], [], [], []
        for x, y in tqdm(tr_loader, total=len(tr_loader)):
            with torch.no_grad():
                _xtr.append(
                    model.encode(x.to('cuda')).cpu())
                _ytr.append(y)
        _xtr = torch.cat(_xtr, dim=0)
        _ytr = torch.cat(_ytr, dim=0)

        for x, y in tqdm(ts_loader, total=len(ts_loader)):
            with torch.no_grad():
                _xts.append(
                    model.encode(x.to('cuda')).cpu())
                _yts.append(y)
        _xts = torch.cat(_xts, dim=0)
        _yts = torch.cat(_yts, dim=0)

        dataset['sources'][domain] = (_xtr, _ytr, _xts, _yts)
    tr_loader = torch.utils.data.DataLoader(
        ImagesDataset(
                root=args.root,
                dataset_name=args.benchmark,
                domains=[args.tgt, ],
                transform=test_transforms,
                train=True,
                test=False,
                multi_source=False
        ),
        batch_size=batch_size,
        shuffle=False)
    ts_loader = torch.utils.data.DataLoader(
        ImagesDataset(
                root=args.root,
                dataset_name=args.benchmark,
                domains=[args.tgt, ],
                transform=test_transforms,
                train=False,
                test=True,
                multi_source=False
        ),
        batch_size=batch_size,
        shuffle=False)
    _xtr, _xts, _ytr, _yts = [], [], [], []
    for x, y in tqdm(tr_loader, total=len(tr_loader)):
        with torch.no_grad():
            _xtr.append(
                model.encode(x.to('cuda')).cpu())
            _ytr.append(y)
    _xtr = torch.cat(_xtr, dim=0)
    _ytr = torch.cat(_ytr, dim=0)

    for x, y in tqdm(ts_loader, total=len(ts_loader)):
        with torch.no_grad():
            _xts.append(
                model.encode(x.to('cuda')).cpu())
            _yts.append(y)
    _xts = torch.cat(_xts, dim=0)
    _yts = torch.cat(_yts, dim=0)
    dataset['target'] = (_xtr, _ytr, _xts, _yts)

    with open(
            os.path.join(args.out_path, f"{args.benchmark}_"
                         f"{args.tgt}_{args.backbone}_{args.size}.pkl"),
            "wb") as f:
        f.write(pickle.dumps(dataset))
