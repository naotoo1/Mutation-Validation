"""Implementation of Mutation Validation Training of tabular dataset for Learning Vector Quantization"""

import logging
import os
import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import prototorch as pt
import prototorch.models as ps
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.warnings import PossibleUserWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader
from mutate_labels import MutatedValidation
from mutated_validation import MutatedValidationScore, TrainRun, EvaluationMetricsType
from dataset import DATA


torch.set_float32_matmul_precision(precision="high")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class HyperParameterSearch(str, Enum):
    TRUE = "optimal"
    FALSE = "non-optimal"


class SavedModelUpdate(str, Enum):
    TRUE = "update"
    FALSE = "keine-update"


class ValidationType(str, Enum):
    CROSSVALIDATION = "cv"
    MUTATEDVALIDATION = "mv"
    HOLDOUT = "ho"


class LVQ(str, Enum):
    GLVQ = "glvq"
    GMLVQ = "gmlvq"
    GTLVQ = "gtlvq"
    RSLVQ = "rslvq"


@dataclass
class TrainModelSummary:
    selected_model_evaluation_metrics_scores: list[float | int]


@dataclass
class TransformedData:
    data: np.ndarray
    labels: np.ndarray
    evaluation_metric: list[str]
    scoring: dict
    selected_features: list[str]
    rejected_features: list[str]


@dataclass
class PerturbedData:
    data: torch.Tensor


@dataclass
class TensorSet:
    data: torch.Tensor
    labels: torch.Tensor


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
    log: bool = False,
) -> None:
    report = [{validation: evaluation_metric_scores}]
    match log:
        case True:
            return logging.info("%s:%s", model_name, report)
        case False:
            print(f"{model_name} , {report}")


def train_hold_out(
    input_data: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    optimal_search: str,
    save_model: bool = False,
) -> TrainModelSummary:
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, labels, test_size=0.3, random_state=4
    )

    x_input = torch.from_numpy(X_train).to(torch.float32)
    y_label = torch.from_numpy(y_train).to(torch.float32)
    x_input_test = torch.from_numpy(X_test).to(torch.float32)
    y_label_test = torch.from_numpy(y_test).to(torch.float32)

    train_ds = data.TensorDataset(x_input, y_label)

    train_loader = DataLoader(train_ds, batch_size=128, num_workers=2)

    model = specified_lvq(model_name, train_ds)

    trainer = model_trainer(optimal_search)

    trainer.fit(model, train_loader)

    outputs = model.predict(torch.Tensor(x_input_test))

    if save_model:
        save_train_model(
            saved_model_dir="./saved_models",
            model_name=model_name + "_" + ValidationType.HOLDOUT,
            estimator=model,
            scoring_metric=EvaluationMetricsType.ACCURACY,
        )

    return TrainModelSummary([round(accuracy_score(y_label_test, outputs),4)])  # type: ignore


kfold = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
mean_scores = []


def train_model_cv(
    input_data: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    optimal_search: str,
    save_model: bool = False,
) -> TrainModelSummary:
    metric_scores = []

    x_input = torch.from_numpy(input_data).to(torch.float32)
    y_label = torch.from_numpy(labels).to(torch.float32)

    for fold, (train_index, test_index) in enumerate(kfold.split(x_input, y_label)):  # type: ignore
        train_ds = data.TensorDataset(x_input[train_index], y_label[train_index])
        train_loader = DataLoader(train_ds, batch_size=128, num_workers=2)

        model = specified_lvq(model_name, train_ds)

        model.apply(reset_weights)

        trainer = model_trainer(optimal_search)

        trainer.fit(model, train_loader)

        outputs = model.predict(torch.Tensor(x_input[test_index]))
        metric_scores.append(accuracy_score(y_label[test_index], outputs))

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
    input_data: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    optimal_search: str,
    save_model: bool = False,
) -> TrainModelSummary:
    x_input = torch.from_numpy(input_data).to(torch.float32)
    y_label = torch.from_numpy(labels).to(torch.float32)

    train_ds = data.TensorDataset(x_input, y_label)

    train_loader = DataLoader(train_ds, batch_size=128, num_workers=4)

    model = specified_lvq(model_name, train_ds)

    model_mv = specified_lvq(model_name, train_ds)

    model.apply(reset_weights)

    mutated_validation = MutatedValidation(
        labels=labels.astype(np.int64),
        perturbation_ratio=0.2,
        perturbation_distribution="balanced",
    )

    mutate_list = mutated_validation.get_mutated_label_list
    mutated_labels = torch.from_numpy(mutate_list).to(torch.float32)

    train_ds_mutated = data.TensorDataset(x_input, mutated_labels)
    train_loader_mutated = DataLoader(train_ds_mutated, batch_size=128, num_workers=2)

    results, mv_score = [], []
    for train_runs in range(2):
        match train_runs:
            case TrainRun.ORIGINAL:
                trainer = model_trainer(optimal_search)
                trainer.fit(model, train_loader)

                results.append(model.predict(x_input))
                if save_model:
                    save_train_model(
                        saved_model_dir="./saved_models",
                        model_name=model_name,
                        estimator=model,
                        scoring_metric=EvaluationMetricsType.ACCURACY,
                    )

            case TrainRun.MUTATED:
                model_mv.apply(reset_weights)
                trainer = model_trainer(optimal_search)
                trainer.fit(model_mv, train_loader_mutated)
                results.append(model_mv.predict(x_input))

                mv_scorer = MutatedValidationScore(
                    mutated_labels=mutated_validation,
                    mutate=mutate_list,
                    original_training_predicted_labels=results[0],
                    mutated_training_predicted_labels=results[1],
                    evaluation_metric=EvaluationMetricsType.ACCURACY,
                )
                mv_score.append(round(mv_scorer.get_mv_score,4))

    return TrainModelSummary(mv_score)


def gmlvq(train_ds: data.TensorDataset) -> ps.GMLVQ:
    hparams = dict(
        input_dim=2,
        latent_dim=2,
        distribution={"num_classes": 2, "per_class": 5},
        proto_lr=0.01,
        bb_lr=0.01,
    )

    return ps.GMLVQ(
        hparams,
        prototypes_initializer=pt.initializers.SMCI(train_ds, noise=0.1),
    )


def glvq(train_ds: data.TensorDataset) -> ps.GLVQ:
    hparams = dict(
        distribution={"num_classes": 2, "per_class": 1},
        lr=0.01,
    )
    return ps.GLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds, noise=0.1),
    )


def gtlvq(train_ds: data.TensorDataset) -> ps.GTLVQ:
    hparams = dict(
        distribution={"num_classes": 2, "per_class": 1},
        input_dim=2,
        latent_dim=1,
        lr=0.01,
    )

    return ps.GTLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SMCI(train_ds, noise=0.1),
    )


def rslvq(train_ds: data.TensorDataset) -> ps.RSLVQ:
    hparams = dict(
        distribution={"num_classes": 2, "per_class": 1},
        proto_lr=0.5,  # 0.05
        lambd=0.9,  # 0.1,
        variance=2,
        input_dim=2,
        latent_dim=2,
        bb_lr=0.01,
    )

    return ps.RSLVQ(
        hparams,
        optimizer=torch.optim.Adam,
        prototypes_initializer=pt.initializers.SSCI(train_ds, noise=0.2),
    )


def specified_lvq(
    model: str, train_ds: data.TensorDataset
) -> ps.GLVQ | ps.GMLVQ | ps.GTLVQ | ps.RSLVQ:
    match model:
        case LVQ.GLVQ:
            return glvq(train_ds)
        case LVQ.GMLVQ:
            return gmlvq(train_ds)
        case LVQ.GTLVQ:
            return gtlvq(train_ds)
        case LVQ.RSLVQ:
            return rslvq(train_ds)
        case _:
            raise RuntimeError("specified_lvq: none of the models did match")


def model_trainer(search: str) -> pl.Trainer:  # type: ignore
    match search:
        case HyperParameterSearch.FALSE:
            trainer = pl.Trainer(
                max_epochs=100,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=False,
                detect_anomaly=False,
                enable_model_summary=False,
                accelerator="cpu",
            )
            return trainer
        case HyperParameterSearch.TRUE:
            trainer = pl.Trainer(
                max_epochs=100,
                enable_progress_bar=False,
                enable_checkpointing=False,
                logger=False,
                detect_anomaly=True,
                enable_model_summary=False,
                accelerator="gpu",
            )
            return trainer


def save_train_model(
    *,
    saved_model_dir: str,
    model_name: str,
    estimator: ps.GLVQ | ps.GMLVQ,
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


def get_numpy_as_tensor(input_data: np.ndarray, labels: np.ndarray) -> TensorSet:
    x_input = torch.from_numpy(input_data).to(torch.float32)
    y_label = torch.from_numpy(labels).to(torch.float32)
    return TensorSet(x_input, y_label)


@dataclass(slots=True)
class TM:
    input_data: np.ndarray
    labels: np.ndarray
    model_name: str
    optimal_search: str
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
            self.save_model,
        )

    @property
    def train_cv(self) -> TrainModelSummary:
        return train_model_cv(
            self.input_data,
            self.labels,
            self.model_name,
            self.optimal_search,
            self.save_model,
        )

    @property
    def train_mv(self) -> TrainModelSummary:
        return train_model_by_mv(
            self.input_data,
            self.labels,
            self.model_name,
            self.optimal_search,
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
        input_data=train_data.S_1.input_data,
        labels=train_data.S_1.labels,
        model_name=LVQ.GLVQ,
        optimal_search=HyperParameterSearch.FALSE,
        log=True
    )

    # train and evaluate using Holdout, CV and MV scheme
    EVALUATE = train.train_all
