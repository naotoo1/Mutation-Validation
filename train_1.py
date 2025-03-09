"""Implementation of Mutation Validation Training of MNIST dataset for Learning Vector Quantization"""

import logging
import random
import warnings
from dataclasses import dataclass, field
import os
from pathlib import Path
import numpy as np
import torch
from lightning_fabric.utilities.warnings import PossibleUserWarning
from prototorch.models import PruneLoserPrototypes
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils import data
import prototorch as pt
import prototorch.models as ps
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from mutate_labels import MutatedValidation
from mutated_validation import MutatedValidationScore, TrainRun, EvaluationMetricsType
from train import (
    LVQ,
    ValidationType,
    HyperParameterSearch,
    TrainModelSummary,
)

from dataset import DATA

torch.set_float32_matmul_precision(precision="high")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


Path("./evaluation").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="evaluation/evaluation_metrics_scores_0.2.txt",
    encoding="utf-8",
    filemode="w",
    level=logging.INFO,
)


def evaluation_metric_logs(
    evaluation_metric_scores: list[float | int],
    model_name: str,
    validation: str,
    log: bool = True,
) -> None:
    report = [{validation: evaluation_metric_scores}]
    match log:
        case True:
            return logging.info("%s:%s", model_name, report)
        case False:
            print(f"{model_name} , {report}")


kfold = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
mean_scores = []


def train_hold_out(
    input_data: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    optimal_search: str,
    dynamic_proto_pruning: bool,
    save_model: bool = False,
) -> TrainModelSummary:
    train_set, train_labels = input_data[:1000], labels[:1000]

    test_set, test_labels = input_data[-1000:], labels[-1000:]

    train_ds = data.TensorDataset(train_set, train_labels)

    train_loader = DataLoader(train_ds, batch_size=256, num_workers=2)

    model = specified_lvq(model_name, train_ds, train_loader)

    trainer = model_trainer(optimal_search, dynamic_proto_pruning)

    trainer.fit(model, train_loader)

    outputs = model.predict(torch.Tensor(test_set))

    if save_model:
        save_train_model(
            saved_model_dir="./saved_models",
            model_name=model_name + "_" + ValidationType.HOLDOUT,
            estimator=model,
            scoring_metric=EvaluationMetricsType.ACCURACY,
        )

    return TrainModelSummary([accuracy_score(test_labels, outputs)])  # type: ignore


def train_model_cv(
    input_data: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    optimal_search: str,
    dynamic_proto_pruning: bool,
    save_model: bool = False,
) -> TrainModelSummary:
    metric_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(input_data, labels)):  # type: ignore
        train_ds = data.TensorDataset(input_data[train_index], labels[train_index])
        train_loader = DataLoader(train_ds, batch_size=128, num_workers=2)

        model = specified_lvq(model_name, train_ds, train_loader)

        model.apply(reset_weights)

        trainer = model_trainer(optimal_search, dynamic_proto_pruning)

        trainer.fit(model, train_loader)

        outputs = model.predict(torch.Tensor(input_data[test_index]))

        metric_scores.append(accuracy_score(labels[test_index], outputs))

        if fold == 4:
            if save_model:
                save_train_model(
                    saved_model_dir="./saved_models",
                    model_name=model_name + "_" + ValidationType.HOLDOUT,
                    estimator=model,
                    scoring_metric=EvaluationMetricsType.ACCURACY,
                )

    mean_scores.append(
        round((float(np.mean(metric_scores))),4)
    )

    return TrainModelSummary(mean_scores)


def train_model_by_mv(
    input_data: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    optimal_search: str,
    dynamic_proto_pruning: bool = False,
    save_model: bool = False,
) -> TrainModelSummary:
    train_ds = data.TensorDataset(input_data, labels)

    train_loader = DataLoader(train_ds, batch_size=256, num_workers=2)

    model = specified_lvq(model_name, train_ds, train_loader)

    model_mv = specified_lvq(model_name, train_ds, train_loader)

    model.apply(reset_weights)

    label = labels.cpu().detach().numpy()

    mutated_validation = MutatedValidation(
        labels=label.astype(np.int64),
        perturbation_ratio=0.2,
        perturbation_distribution="balanced",
    )

    mutate_list = mutated_validation.get_mutated_label_list
    mutated_labels = torch.from_numpy(mutate_list).to(torch.float32)

    train_ds_mutated = data.TensorDataset(input_data, mutated_labels)
    train_loader_mutated = DataLoader(train_ds_mutated, batch_size=256, num_workers=2)

    results, mv_score = [], []
    for train_runs in range(2):
        match train_runs:
            case TrainRun.ORIGINAL:
                trainer = model_trainer(
                    search=optimal_search, dynamic_proto_pruning=dynamic_proto_pruning
                )
                trainer.fit(model, train_loader)

                results.append(model.predict(input_data))
                if save_model:
                    save_train_model(
                        saved_model_dir="./saved_models",
                        model_name=model_name,
                        estimator=model,
                        scoring_metric=EvaluationMetricsType.ACCURACY,
                    )

            case TrainRun.MUTATED:
                model_mv.apply(reset_weights)
                trainer = model_trainer(
                    search=optimal_search, dynamic_proto_pruning=dynamic_proto_pruning
                )
                trainer.fit(model_mv, train_loader_mutated)
                results.append(model_mv.predict(input_data))

                mv_scorer = MutatedValidationScore(
                    mutated_labels=mutated_validation,
                    mutate=mutate_list,
                    original_training_predicted_labels=results[0],
                    mutated_training_predicted_labels=results[1],
                    evaluation_metric=EvaluationMetricsType.ACCURACY,
                )
                mv_score.append(round(mv_scorer.get_mv_score,4))

    return TrainModelSummary(mv_score)


def gtlvq(
    train_ds: data.TensorDataset,
    train_loader: torch.utils.data.DataLoader,  # type: ignore
) -> ps.ImageGTLVQ:
    hparams = dict(
        input_dim=28 * 28,
        latent_dim=28,
        distribution={"num_classes": 10, "per_class": 1},
        proto_lr=0.01,
        bb_lr=0.01,
    )

    return ps.ImageGTLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
        # Use one batch of data for subspace initiator.
        omega_initializer=pt.initializers.PCALinearTransformInitializer(
            next(iter(train_loader))[0].reshape(256, 28 * 28)
        ),
    )


def glvq(train_ds: data.TensorDataset) -> ps.ImageGLVQ:
    hparams = dict(
        distribution={"num_classes": 10, "per_class": 1},
        lr=0.01,
    )
    return ps.ImageGLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
    )


def gmlvq(train_ds: data.TensorDataset) -> ps.ImageGMLVQ:
    hparams = dict(
        input_dim=28 * 28,
        latent_dim=28 * 28,
        distribution={"num_classes": 10, "per_class": 1},
        proto_lr=0.01,
        bb_lr=0.01,
    )

    return ps.ImageGMLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds),
    )


def specified_lvq(
    model: str,
    train_ds: data.TensorDataset,
    train_loader: torch.utils.data.DataLoader,  # type: ignore
) -> ps.ImageGLVQ | ps.ImageGMLVQ | ps.ImageGTLVQ:
    match model:
        case LVQ.GLVQ:
            return glvq(train_ds)
        case LVQ.GMLVQ:
            return gmlvq(train_ds)
        case LVQ.GTLVQ:
            return gtlvq(train_ds, train_loader)
        case _:
            raise RuntimeError("specified_lvq: none of the models did match")


def model_trainer(search: str, dynamic_proto_pruning: bool) -> pl.Trainer:
    if search == HyperParameterSearch.FALSE and dynamic_proto_pruning is True:
        trainer = pl.Trainer(
            callbacks=[
                PruneLoserPrototypes(
                    threshold=0.01,
                    idle_epochs=1,
                    prune_quota_per_epoch=10,
                    frequency=1,
                    verbose=True,
                ),
                EarlyStopping(
                    monitor="train_loss",
                    min_delta=0.001,
                    patience=15,
                    mode="min",
                    check_on_train_epoch_end=True,
                ),
            ],
            max_epochs=1000,
            log_every_n_steps=1,
            detect_anomaly=True,
            accelerator="cpu",
        )
        return trainer
    if HyperParameterSearch.TRUE:
        trainer = pl.Trainer(
            max_epochs=3,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            detect_anomaly=True,
            enable_model_summary=False,
            accelerator="gpu",
        )
        return trainer
    else:
        raise RuntimeError("model_trainer:none of the search marches")


def save_train_model(
    *,
    saved_model_dir: str,
    model_name: str,
    estimator: ps.ImageGLVQ | ps.ImageGMLVQ | ps.ImageGTLVQ,
    scoring_metric: str,
):
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            estimator,
            os.path.join(
                saved_model_dir,
                model_name + scoring_metric + ".pt",
            ),
        )
    except AttributeError:
        pass


@dataclass(slots=True)
class TM:
    input_data: torch.Tensor
    labels: torch.Tensor
    model_name: str
    optimal_search: str
    dynamic_proto_pruning: bool = False
    save_model: bool = False
    log: bool =True
    summary_metric_list: list = field(default_factory=lambda: [])

    @property
    def train_ho(self) -> TrainModelSummary:
        return train_hold_out(
            self.input_data,
            self.labels,
            self.model_name,
            self.optimal_search,
            self.dynamic_proto_pruning,
            self.save_model,
        )

    @property
    def train_cv(self) -> TrainModelSummary:
        return train_model_cv(
            self.input_data,
            self.labels,
            self.model_name,
            self.optimal_search,
            self.dynamic_proto_pruning,
            self.save_model,
        )

    @property
    def train_mv(self) -> TrainModelSummary:
        return train_model_by_mv(
            self.input_data,
            self.labels,
            self.model_name,
            self.optimal_search,
            self.dynamic_proto_pruning,
            self.save_model,
        )

    @property
    def train_all(self) -> logging.INFO:  # type: ignore
        self.summary_metric_list.append(
            self.train_ho.selected_model_evaluation_metrics_scores
        )
        self.summary_metric_list.append(
            self.train_cv.selected_model_evaluation_metrics_scores
        )
        self.summary_metric_list.append(
            self.train_mv.selected_model_evaluation_metrics_scores
        )

        results = list(np.array(self.summary_metric_list).ravel())
        return evaluation_metric_logs(
            [f"HO: {results[0]}, CV: {results[1]}, MV: {results[2]}"],
            self.model_name,
            EvaluationMetricsType.ACCURACY.value,
            self.log
        )


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


if __name__ == "__main__":
    seed_everything(seed=4)
    train_data = DATA(random=4)
    train = TM(
        input_data=train_data.mnist.input_data.float(),
        labels=train_data.mnist.labels,
        model_name=LVQ.GMLVQ,
        optimal_search=HyperParameterSearch.FALSE,
        log=True
    )

    # train and evaluate using Holdout, CV and MV scheme
    EVALUATE = train.train_all
