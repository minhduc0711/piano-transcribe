from argparse import ArgumentParser

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier

from src.data.audio import onf_transform
from src.data.data_modules import MAPSDataModule
from src.eval import compute_frame_metrics


def train_and_eval_model(clf, dm, max_steps, num_iters=50):
    dm.set_max_steps(max_steps)
    mid_window = max_steps // 2
    for it in tqdm(range(num_iters), desc="Epoch", leave=False):
        for i, batch in tqdm(
            enumerate(dm.train_dataloader()),
            total=len(dm.train_dataloader()),
            leave=False,
            desc="Batch",
        ):
            batch_size = batch["audio"].shape[0]
            feats = batch["audio"].numpy().reshape(batch_size, -1)
            labels = batch["frames"].numpy()[:, mid_window, :].astype(np.int)
            classes = (
                [np.array([0, 1]) for i in range(labels.shape[-1])] if i == 0 else None
            )
            clf.partial_fit(feats, labels, classes=classes)

    # evaluate metrics on full sequences
    print("Evaluating the model...")
    sample_metrics = []
    for batch in tqdm(dm.test_dataloader(), desc="Test sample", leave=False):
        seqs = batch["audio"].numpy().squeeze()
        num_steps, in_feats = seqs.shape
        # zero-pad at beginning and end of sequences
        pad_size = max_steps // 2
        pad = np.zeros((pad_size, in_feats))
        seqs = np.vstack((pad, seqs, pad))
        # accumulate windows of size max_steps
        windows = []
        for i in range(seqs.shape[0] - max_steps + 1):
            windows.append(seqs[i: i + max_steps])
        windows = np.array(windows).reshape(num_steps, -1)

        y_pred = clf.predict(windows).squeeze()
        labels = batch["frames"].numpy().squeeze()
        sample_metrics.append(compute_frame_metrics(y_pred, labels))

    final_metrics = defaultdict(lambda: defaultdict(lambda: "N/A"))
    for metric_type in sample_metrics[0].keys():
        for cls_metric in sample_metrics[0][metric_type]:
            vals = [metric[metric_type][cls_metric] for metric in sample_metrics]
            final_metrics[metric_type][
                cls_metric
            ] = f"{np.mean(vals):.4f} \u00B1 {np.std(vals):.4f}"
    return final_metrics


parser = ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

dm = MAPSDataModule(
    batch_size=4,
    sample_rate=16000,
    max_steps=1,
    audio_transform=onf_transform,
    hop_length=512,
    lazy_loading=False,
    debug=args.debug,
)
dm.setup()

table = []
for max_steps in [1, 5]:
    for clf in [SGDClassifier(loss="log"),
                SGDClassifier(loss="hinge")]:
        if clf.loss == "log":
            clf_name = "log_reg"
        elif clf.loss == "hinge":
            clf_name = "svm"
        print(f"Training {clf_name} with max_steps={max_steps}")

        clf_wrapper = MultiOutputClassifier(clf)
        metrics = train_and_eval_model(clf_wrapper, dm, max_steps,
                                       num_iters=100)
        frame_metrics = metrics["frame"]
        table.append([
            f"{clf_name} (max_steps={max_steps})",
            frame_metrics["precision"],
            frame_metrics["recall"],
            frame_metrics["f1"]
        ])
print(tabulate(table, headers=["model", "precision", "recall", "f1"]))
