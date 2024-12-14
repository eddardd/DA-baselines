import os
import torch
import pickle
import argparse
import torchvision

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from da_baselines.architectures import get_vit
from da_baselines.architectures import get_resnet
from da_baselines.architectures import get_resnet_head


parser = argparse.ArgumentParser(
    description='Train baseline on the VisDA benchmark')
parser.add_argument('--root',
                    type=str,
                    help='Root for dataset')
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
                    default=30000)
parser.add_argument('--eval_every',
                    type=int,
                    default=1000)
parser.add_argument('--extract_features',
                    type=str,
                    default='True',
                    help="Whether or not extract features")
parser.add_argument('--save_model',
                    type=str,
                    default='True',
                    help='Wether or not to save the model')
parser.add_argument('--out_path',
                    type=str,
                    default='./features',
                    help="path to save features")
parser.add_argument('--out_path_model',
                    type=str,
                    default='./pretrained',
                    help="path to save the models")
args = parser.parse_args()

n_iter = args.n_iter
batch_size = args.batch_size
learning_rate = args.learning_rate

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.root, 'train'), transform=train_transforms)
n_classes = 12
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)

test_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.root, 'validation'), transform=test_transforms)
n_classes = 12
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


if args.backbone.lower() == 'vit':
    model = get_vit(
        n_layers=args.size,
        name=args.vit_type,
        n_classes=n_classes,
        return_T=False).to('cuda')
    encoder_name = f"vit_encoder_{args.vit_type}_{args.size}_visda.pt"
    clf_name = f"vit_clf_{args.vit_type}_{args.size}_visda.pt"
elif args.backbone.lower() == 'resnet':
    model = get_resnet(
        resnet_size=args.size,
        n_classes=n_classes,
        return_T=False
    ).to('cuda')
    encoder_name = f"resnet_encoder_{args.size}_visda.pt"
    clf_name = f"resnet_clf_{args.size}_visda.pt"
elif args.backbone.lower() == 'resnet-head':
    model = get_resnet_head(
        resnet_size=args.size,
        n_classes=n_classes,
        n_hidden=256,
        return_T=False
    ).to('cuda')
    encoder_name = f"resnet_encoder_{args.size}_visda.pt"
    clf_name = f"resnet_clf_{args.size}_visda.pt"
    head_name = f"resnet_head_{args.size}_visda.pt"
else:
    raise ValueError("Expected backbone to be either vit or "
                     f"resnet, but got {args.backbone.lower()}")

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
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
        with open("./logs/visda/train.csv", 'a') as f:
            f.write(f"{it},{loss.item()}\n")
    except FileNotFoundError:
        print("Error, path './logs/visda/train.csv' does not exist")
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
            with open("./logs/visda/test.csv", 'a') as f:
                f.write(f"{it},{ts_loss},{ts_acc}\n")
        except FileNotFoundError:
            print("Error, path './logs/visda/test.csv' does not exist")
            print("Continuing...")
        model.train()

if args.extract_features.lower() == 'true':
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.root, 'train'), transform=test_transforms)
    n_classes = 12
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.root, 'validation'), transform=test_transforms)
    n_classes = 12
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    print("Extracting features")
    dataset = {
        "Train": None, "Test": None
    }
    model.eval()
    for kind, loader in [('Train', train_loader), ('Test', test_loader)]:
        with torch.no_grad():
            Z, Y = [], []
            for x, y in tqdm(loader):
                Z.append(model.encode(x.to('cuda')).cpu())
                Y.append(y)
            dataset[kind] = (torch.cat(Z, dim=0), torch.cat(Y, dim=0))

    with open(
            os.path.join(args.out_path,
                         f"visda_{args.backbone}_{args.size}.pkl"),
            "wb") as f:
        f.write(pickle.dumps(dataset))

if args.save_model.lower() == 'true':
    print("Saving the model.")
    torch.save(
        model.encoder.state_dict(),
        os.path.join(args.out_path_model, encoder_name))
    torch.save(
        model.clf.state_dict(),
        os.path.join(args.out_path_model, clf_name))
    if args.backbone.lower() == "resnet-head":
        torch.save(
            model.head.state_dict(),
            os.path.join(args.out_path_model, head_name))
