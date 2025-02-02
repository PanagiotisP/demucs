# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction

import torch as th
from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Shift
from .compressed import StemsSet, build_musdb_metadata, get_musdb_tracks
from .parser import get_name, get_parser
from .raw import Rawset
from .tasnet import ConvTasNet, SeqConvTasNet
from .test import evaluate
from .train import train_model, validate_model
from .utils import human_seconds, load_model, save_model, sizeof_fmt


@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None
    bad_epochs: int = 0


def main():
    parser = get_parser()
    args = parser.parse_args()
    name = get_name(parser, args)
    print(f"Experiment {name}")

    if args.musdb is None and args.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.json"
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    th.manual_seed(args.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    checkpoint = args.checkpoints / f"{name}.th"
    checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    if args.restart and checkpoint.exists():
        checkpoint.unlink()

    if args.test:
        args.epochs = 1
        args.repeat = 0
        model = load_model(args.models / args.test)
    else:
        if args.seq:
            model = SeqConvTasNet(audio_channels=args.audio_channels,
                                  stacks=args.stacks,
                                  pad=args.pad,
                                  band_num=args.band_num,
                                  skip=args.skip,
                                  racks=args.racks,
                                  hidden_dim=args.hidden_dim,
                                  bottle_dim=args.bottle_dim,
                                  enc_dim=args.enc_dim,
                                  instr_num=args.instr_num,
                                  transform_window=args.transform_window,
                                  dwt=args.dwt,
                                  learnable=args.learnable,
                                  samplerate=args.sr)
        else:
            model = ConvTasNet(audio_channels=args.audio_channels,
                               stacks=args.stacks,
                               pad=args.pad,
                               band_num=args.band_num,
                               dilation_split=args.dilation_split,
                               skip=args.skip,
                               racks=args.racks,
                               hidden_dim=args.hidden_dim,
                               bottle_dim=args.bottle_dim,
                               enc_dim=args.enc_dim,
                               instr_num=args.instr_num,
                               transform_window=args.transform_window,
                               dwt=args.dwt,
                               samplerate=args.sr)
    model.to(device)
    if args.show:
        print(model)
        print('parameter count: ', str(sum(p.numel() for p in model.parameters())))
        size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
        print(f"Model size {size}")
        return

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    try:
        saved = th.load(checkpoint, map_location='cpu')
    except IOError:
        saved = SavedState()
    else:
        model.load_state_dict(saved.last_state)
        optimizer.load_state_dict(saved.optimizer)

    if args.save_model:
        if args.rank == 0:
            model.to("cpu")
            model.load_state_dict(saved.best_state)
            save_model(model, args.models / f"{name}.th")
        return

    if args.rank == 0:
        done = args.logs / f"{name}.done"
        if done.exists():
            done.unlink()

    if args.augment:
        augment = nn.Sequential(FlipSign(), FlipChannels(), Shift(args.data_stride),
                                Remix(group_size=args.remix_group_size)).to(device)
    else:
        augment = Shift(args.data_stride)

    if args.mse:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = model.valid_length(args.samples)
    print(f"Number of training samples adjusted to {samples}")

    if args.raw:
        train_set = Rawset(args.raw / "train",
                           samples=samples + args.data_stride,
                           channels=args.audio_channels,
                           streams=[0, 1, 2, 3, 4] if args.multi else [0, 1, 2])

        valid_set = Rawset(args.raw / "valid", channels=args.audio_channels)
    else:
        if not args.metadata.is_file() and args.rank == 0:
            build_musdb_metadata(args.metadata, args.musdb, args.workers)
        if args.world_size > 1:
            distributed.barrier()
        metadata = json.load(open(args.metadata))
        duration = Fraction(samples + args.data_stride, args.samplerate)
        stride = Fraction(args.data_stride, args.samplerate)
        train_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="train"),
                             metadata,
                             duration=duration,
                             stride=stride,
                             samplerate=args.samplerate,
                             channels=args.audio_channels)
        valid_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="valid"),
                             metadata,
                             samplerate=args.samplerate,
                             channels=args.audio_channels)

    best_loss = float("inf")
    for epoch, metrics in enumerate(saved.metrics):
        print(f"Epoch {epoch:03d}: "
              f"train={metrics['train']:.8f} "
              f"valid={metrics['valid']:.8f} "
              f"best={metrics['best']:.4f} "
              f"duration={human_seconds(metrics['duration'])}")
        best_loss = metrics['best']

    if args.world_size > 1:
        dmodel = DistributedDataParallel(model,
                                         device_ids=[th.cuda.current_device()],
                                         output_device=th.cuda.current_device())
    else:
        dmodel = model

    bad_epochs = saved.bad_epochs
    for epoch in range(len(saved.metrics), args.epochs):
        if bad_epochs >= 20:
            break
        begin = time.time()
        model.train()
        train_loss = train_model(epoch,
                                 train_set,
                                 dmodel,
                                 criterion,
                                 optimizer,
                                 augment,
                                 batch_size=args.batch_size,
                                 device=device,
                                 repeat=args.repeat,
                                 seed=args.seed,
                                 workers=args.workers,
                                 world_size=args.world_size,
                                 instr_num=args.instr_num)
        model.eval()
        valid_loss = validate_model(epoch,
                                    valid_set,
                                    model,
                                    criterion,
                                    device=device,
                                    rank=args.rank,
                                    split=args.split_valid,
                                    world_size=args.world_size,
                                    instr_num=args.instr_num)

        duration = time.time() - begin
        if valid_loss < best_loss:
            best_loss = valid_loss
            saved.best_state = {
                key: value.to("cpu").clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        saved.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
            "duration": duration,
            "bad_epochs": bad_epochs
        })
        if args.rank == 0:
            json.dump(saved.metrics, open(metrics_path, "w"))

        saved.last_state = model.state_dict()
        saved.optimizer = optimizer.state_dict()
        if args.rank == 0 and not args.test:
            th.save(saved, checkpoint_tmp)
            checkpoint_tmp.rename(checkpoint)

        print(f"Epoch {epoch:03d}: "
              f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} "
              f"duration={human_seconds(duration)}")

    del dmodel
    model.load_state_dict(saved.best_state)
    if args.eval_cpu:
        device = "cpu"
        model.to(device)
    model.eval()
    evaluate(model,
             args.musdb,
             eval_folder,
             rank=args.rank,
             world_size=args.world_size,
             device=device,
             save=args.save,
             split=args.split_valid,
             shifts=args.shifts,
             workers=args.eval_workers,
             samplerate=args.samplerate,
             channels=args.audio_channels)
    model.to("cpu")
    save_model(model, args.models / f"{name}.th")
    if args.rank == 0:
        print("done")
        done.write_text("done")


if __name__ == "__main__":
    main()
