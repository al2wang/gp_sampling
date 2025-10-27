# sampling_diff_energies.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



from bgflow import (
    DoubleWellEnergy,
    GaussianMCMCSampler,
    NormalDistribution,
    BoltzmannGenerator
)
from bgflow.nn import (
    DenseNet, SequentialFlow, CouplingFlow,
    AffineFlow, SplitFlow, InverseFlow,
    SwapFlow, AffineTransformer
)
from bgflow.utils.train import IndexBatchIterator
from bgflow.utils.types import assert_numpy

from hartmann6d import Hartmann6DEnergy



USE_HARTMANN = False
dim = 6 if USE_HARTMANN else 2

# define target energy
if USE_HARTMANN:
    target = Hartmann6DEnergy(dim)
else:
    target = DoubleWellEnergy(dim)




import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--use_hartmann", type=int, default=0)
args = parser.parse_args()

USE_HARTMANN = bool(args.use_hartmann)
dim = 6 if USE_HARTMANN else 2




def plot_energy(energy, extent=(-2.5, 2.5), resolution=100, dim=2):
    # plot 2D energy landscape
    if dim != 2:
        print("skipping 2d plot: dimension != 2")
        return
    xs = torch.meshgrid([torch.linspace(*extent, resolution) for _ in range(2)])
    xs = torch.stack(xs, dim=-1).view(-1, 2)
    xs = torch.cat([
        xs,
        torch.Tensor(xs.shape[0], dim - xs.shape[-1]).zero_()
    ], dim=-1)
    us = energy.energy(xs).view(resolution, resolution)
    us = torch.exp(-us)
    plt.imshow(assert_numpy(us).T, extent=extent * 2)
    plt.title("Energy Landscape")
    plt.show()


def plot_samples(samples, weights=None, range=None):
    # plot 2D histogram of samples
    samples = assert_numpy(samples)
    plt.hist2d(
        samples[:, 0],
        -samples[:, 1],
        weights=assert_numpy(weights) if weights is not None else weights,
        bins=100,
        norm=mpl.colors.LogNorm(),
        range=range
    )
    plt.title("Samples")
    plt.show()


prior = NormalDistribution(dim)

# define RealNVP-style flow
layers = []
layers.append(SplitFlow(dim // 2))

n_coupling_layers = args.n_layers

for _ in range(n_coupling_layers):
    layers.append(SwapFlow())
    layers.append(CouplingFlow(
        AffineTransformer(
            shift_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU()),
            scale_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU())
        )
    ))
layers.append(InverseFlow(SplitFlow(dim // 2)))
flow = SequentialFlow(layers)

bg = BoltzmannGenerator(prior, flow, target)

if not USE_HARTMANN:
    init_state = torch.Tensor([[-2, 0], [2, 0]])
    init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim - 2).normal_()], dim=-1)
    target_sampler = GaussianMCMCSampler(target, init_state=init_state)
    data = target_sampler.sample(50000)
else:
    # for Hartmann, sample random points in [0,1]^6
    data = torch.rand(50000, dim)


# training
class LossReporter:
    # simple loss logger and plotter
    def __init__(self, *labels):
        self._labels = labels
        self._raw = [[] for _ in labels]

    def report(self, *losses):
        for i, loss in enumerate(losses):
            self._raw[i].append(assert_numpy(loss))

    def plot(self, n_smooth=10):
        fig, axes = plt.subplots(len(self._labels), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.set_size_inches((8, 4 * len(self._labels)), forward=True)
        for i, (label, raw, axis) in enumerate(zip(self._labels, self._raw, axes)):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            smoothed = np.convolve(raw, kernel, mode="valid")
            axis.plot(smoothed)
            axis.set_ylabel(label)
            if i == len(self._labels) - 1:
                axis.set_xlabel("Iteration")
        plt.show()


# phase 1 - maximum likelihood training
n_batch = args.batch_size
batch_iter = IndexBatchIterator(len(data), n_batch)

optim = torch.optim.Adam(bg.parameters(), lr=args.lr)

n_epochs = 5
n_report_steps = 50
reporter = LossReporter("NLL")

print("Starting ML training...")
for epoch in range(n_epochs):
    for it, idxs in enumerate(batch_iter):
        batch = data[idxs]
        optim.zero_grad()
        nll = bg.energy(batch).mean()
        nll.backward()
        reporter.report(nll)
        optim.step()
        if it % n_report_steps == 0:
            print(f"\repoch: {epoch}, iter: {it}/{len(batch_iter)}, NLL: {nll.item():.4f}", end="")
print("\nDone ML phase.")

reporter.plot()

# phase 2: mixture of NLL and KL
n_kl_samples = 128
n_batch = 128
batch_iter = IndexBatchIterator(len(data), n_batch)
optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
n_epochs = 5
n_report_steps = 50
lambdas = torch.linspace(1., 0.5, n_epochs)
reporter2 = LossReporter("NLL", "KLL")

print("Starting ML+KL phase...")
for epoch, lamb in enumerate(lambdas):
    for it, idxs in enumerate(batch_iter):
        batch = data[idxs]
        optim.zero_grad()
        nll = bg.energy(batch).mean()
        (lamb * nll).backward()
        kll = bg.kldiv(n_kl_samples).mean()
        ((1. - lamb) * kll).backward()
        reporter2.report(nll, kll)
        optim.step()
        if it % n_report_steps == 0:
            print(f"\repoch: {epoch}, iter: {it}/{len(batch_iter)}, lambda: {lamb:.3f}, NLL: {nll.item():.4f}, KLL: {kll.item():.4f}", end="")
print("\nDone ML+KL phase.")
reporter2.plot()

