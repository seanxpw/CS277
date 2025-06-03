import nni
import torch
import click
torch.set_float32_matmul_precision('medium')
from nni.nas.evaluator.pytorch import DataLoader
from cs277_dataset import CS277Dataset
from pathlib import Path
from cs277_graph_cov_space import NasBench201 as DartsSpace
from nni.nas.evaluator.pytorch import Classification
import torchmetrics

fast_dev_run = False


def nas(train_data):
    model_space = DartsSpace()

    import numpy as np
    from torch.utils.data import SubsetRandomSampler

    num_samples = len(train_data)
    indices = np.random.permutation(num_samples)
    split = num_samples // 2

    search_train_loader = DataLoader(
        train_data, batch_size=64, num_workers=6,
        sampler=SubsetRandomSampler(indices[:split]),
    )

    search_valid_loader = DataLoader(
        train_data, batch_size=64, num_workers=6,
        sampler=SubsetRandomSampler(indices[split:]),
    )

    evaluator = Classification(
        torch.nn.BCEWithLogitsLoss,
        learning_rate=1e-3,
        weight_decay=1e-4,
        train_dataloaders=search_train_loader,
        val_dataloaders=search_valid_loader,
        max_epochs=33,
        num_classes=20,
        # gpus=1,
        fast_dev_run=fast_dev_run,
        metrics={
            'acc': torchmetrics.Accuracy('multilabel', num_labels=20)
        }
    )

    # %%
    #
    # Strategy
    # ^^^^^^^^
    #
    # We will use `DARTS`_ (Differentiable ARchiTecture Search) as the search strategy to explore the model space.
    # :class:`~nni.nas.strategy.DARTS` strategy belongs to the category of :ref:`one-shot strategy <one-shot-nas>`.
    # The fundamental differences between One-shot strategies and :ref:`multi-trial strategies <multi-trial-nas>` is that,
    # one-shot strategy combines search with model training into a single run.
    # Compared to multi-trial strategies, one-shot NAS doesn't need to iteratively spawn new trials (i.e., models),
    # and thus saves the excessive cost of model training.
    #
    # .. note::
    #
    #    It's worth mentioning that one-shot NAS also suffers from multiple drawbacks despite its computational efficiency.
    #    We recommend
    #    `Weight-Sharing Neural Architecture Search: A Battle to Shrink the Optimization Gap <https://arxiv.org/abs/2008.01475>`__
    #    and
    #    `How Does Supernet Help in Neural Architecture Search? <https://arxiv.org/abs/2010.08219>`__ for interested readers.
    #
    # :class:`~nni.nas.strategy.DARTS` strategy is provided as one of NNI's :doc:`built-in search strategies </nas/exploration_strategy>`.
    # Using it can be as simple as one line of code.

    from nni.nas.strategy import GumbelDARTS as DartsStrategy

    strategy = DartsStrategy()

    # %%
    #
    # .. tip:: The ``DartsStrategy`` here can be replaced by any search strategies, even multi-trial strategies.
    #
    # If you want to know how DARTS strategy works, here is a brief version.
    # Under the hood, DARTS converts the cell into a densely connected graph, and put operators on edges (see the following figure).
    # Since the operators are not decided yet, every edge is a weighted mixture of multiple operators (multiple color in the figure).
    # DARTS then learns to assign the optimal "color" for each edge during the network training.
    # It finally selects one "color" for each edge, and drops redundant edges.
    # The weights on the edges are called *architecture weights*.
    #
    # .. image:: ../../img/darts_illustration.png
    #
    # .. tip:: It's NOT reflected in the figure that, for DARTS model space, exactly two inputs are kept for every node.
    #
    # Launch experiment
    # ^^^^^^^^^^^^^^^^^
    #
    # We then come to the step of launching the experiment.
    # This step is similar to what we have done in the :doc:`beginner tutorial <hello_nas>`.

    from nni.nas.experiment import NasExperiment

    experiment = NasExperiment(model_space, evaluator, strategy)
    experiment.run()

    # %%
    #
    # .. tip::
    #
    #    The search process can be visualized with tensorboard. For example::
    #
    #        tensorboard --logdir=./lightning_logs
    #
    #    Then, open the browser and go to http://localhost:6006/ to monitor the search process.
    #
    #    .. image:: ../../img/darts_search_process.png
    #
    # We can then retrieve the best model found by the strategy with ``export_top_models``.
    # Here, the retrieved model is a dict (called *architecture dict*) describing the selected normal cell and reduction cell.

    exported_arch = experiment.export_top_models(formatter='dict')
    with open('exported_arch.json', 'w') as f:
        import json
        json.dump(exported_arch, f, indent=4)
    return exported_arch[0]


def train(exported_arch, train_data, valid_data):
    valid_loader = DataLoader(valid_data, batch_size=100, num_workers=5)
    # %%
    #
    # Retrain the searched model
    # --------------------------
    #
    # What we have got in the last step, is only a cell structure.
    # To get a final usable model with trained weights, we need to construct a real model based on this structure,
    # and then fully train it.
    #
    # To construct a fixed model based on the architecture dict exported from the experiment,
    # we can use :func:`nni.nas.space.model_context`. Under the with-context, we will creating a fixed model based on ``exported_arch``,
    # instead of creating a space.

    from nni.nas.space import model_context

    with model_context(exported_arch):
        final_model = DartsSpace()

    # %%
    #
    # We then train the model on full CIFAR-10 training dataset, and evaluate it on the original CIFAR-10 validation dataset.

    train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data

    # %%
    #
    # We must create a new evaluator here because a different data split is used.
    # Also, we should avoid the underlying pytorch-lightning implementation of :class:`~nni.nas.evaluator.pytorch.Classification`
    # evaluator from loading the wrong checkpoint.

    max_epochs = 100

    evaluator = Classification(
        torch.nn.BCEWithLogitsLoss,
        learning_rate=1e-3,
        weight_decay=1e-4,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        max_epochs=max_epochs,
        num_classes=20,
        # gpus=1,
        export_onnx=False,  # Disable ONNX export for this experiment
        fast_dev_run=fast_dev_run,  # Should be false for fully training
        metrics={
            'acc': torchmetrics.Accuracy('multilabel', num_labels=20)
        }
    )

    evaluator.fit(final_model)

@click.command()
@click.option('--dev', 'dev', default=False, is_flag=True, help='Only run one epoch for nas and training')
@click.option('--data-set', 'root_dir', required=True, help='Path to the dataset. For example, "/app/vertex_cover_btute_force_20250602_113052"', type=click.Path(True, False))
def main(dev, root_dir):
    root_dir = str(root_dir)
    if dev:
        global fast_dev_run
        fast_dev_run = True
    train_data = nni.trace(CS277Dataset)(train=True,
                                         root_dir=root_dir,
                                         transform=None)
    valid_data = nni.trace(CS277Dataset)(train=False,
                                         root_dir=root_dir,
                                         transform=None)
    import json
    exported_arch_json = Path('exported_arch.json')
    if exported_arch_json.exists():
        with open(exported_arch_json, 'r') as f:
            exported_arch = json.load(f)[0]
    else:
        exported_arch = nas(train_data)
    train(exported_arch, train_data, valid_data)  # Train the model with the exported architecture


if __name__ == '__main__':
    main()
