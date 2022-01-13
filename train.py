import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets
from util import NumpyDataset, VisualCallback
from models import VanillaFlow, HTSFlow, HTCFlow, TDFlow, COMETFlow


if __name__ == "__main__":

    wandb.init(project="comet-flows", entity="andrewmcdonald")
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", default="vanilla", type=str, help="Model to train",
                        choices=[
                            "vanilla",
                            "hts-1", "hts-2", "hts-4",
                            "htc-10", "htc-05", "htc-01",
                            "td",
                            "comet-10", "comet-05", "comet-01"
                        ])
    parser.add_argument("--data", default="power", type=str, help="Dataset to train on",
                        choices=[
                            "artificial",
                            "bsds300",
                            "cifar10",
                            "climdex",
                            "gas",
                            "hepmass",
                            "miniboone",
                            "mnist",
                            "power"
                        ])
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size to train with")
    parser.add_argument("--hidden_ds", default=(64, 64, 64), type=tuple, help="Hidden dimensions in coupling NN")
    parser.add_argument("--n_samples", default=1000, type=tuple, help="Number of samples to generate")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    args = parser.parse_args()

    # configure data
    data = None
    if args.data == "artificial":
        raise NotImplementedError()
    elif args.data == "bsds300":
        data = datasets.BSDS300()
    elif args.data == "cifar10":
        data = datasets.CIFAR10()
    elif args.data == "climdex":
        raise NotImplementedError()
    elif args.data == "gas":
        data = datasets.GAS()
    elif args.data == "hepmass":
        data = datasets.HEPMASS()
    elif args.data == "miniboone":
        data = datasets.MINIBOONE()
    elif args.data == "mnist":
        data = datasets.MNIST()
    elif args.data == "power":
        data = datasets.POWER()

    # configure dataloaders
    train_dataset = NumpyDataset(data.trn.x)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataset = NumpyDataset(data.val.x)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    model = None
    d = data.trn.x.shape[1]
    if args.model == "vanilla":
        model = VanillaFlow(d, args.hidden_ds, args.lr)
    elif args.model[:3] == "hts":
        dof = int(args.model.split("-")[-1])
        model = HTSFlow(d, args.hidden_ds, args.lr, args.dof)
    elif args.model[:3] == "htc":
        pass
    elif args.model[:2] == "td":
        model = TDFlow(d, args.hidden_ds, args.lr)
    elif args.model[:5] == "comet":
        pass

    # wandb logging
    wandb.watch(model, log="all", log_freq=10)
    wandb_logger = pl.loggers.WandbLogger(project="comet-flows")

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="v_loss"))
    trainer.callbacks.append(VisualCallback(n_samples=args.n_samples, color=data.color, image_size=data.image_size))

    trainer.fit(model, train_dataloader, val_dataloader)
