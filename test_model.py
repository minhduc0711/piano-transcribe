from collections import defaultdict

from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from src.data.data_modules import MAPSDataModule
from src.eval import compute_note_metrics
from src.models.onsets_and_frames import OnsetsAndFrames


# testing over full audio files
dm = MAPSDataModule(batch_size=1,
                    sample_rate=16000,
                    max_steps=None,
                    lazy_loading=True,
                    debug=True)
dm.setup(stage="test")

model = OnsetsAndFrames.load_from_checkpoint("models/onf-MAPS-epoch=379-valid_loss=0.08.ckpt",
                                             in_feats=229)
model.eval()

# a bit different from the validation loop, which partially works with batches
sample_metrics = []  # each elem is a metric dict for 1 sample
for batch in tqdm(dm.test_dataloader(), "Test samples"):
    # unpack data from batch (of 1 sample actually)
    audio_feats = batch["audio"]
    onset_true, frame_true, velocity_true = (
        batch["onsets"],
        batch["frames"],
        batch["velocity"],
    )
    sample_rate = batch["sample_rate"][0].item()
    hop_length = batch["hop_length"][0].item()

    onset_pred, frame_pred, velocity_pred = model(audio_feats)

    p_est, i_est, v_est = model.extract_notes(
        onset_pred.squeeze(),
        frame_pred.squeeze(),
        velocity_pred.squeeze(),
        sample_rate=sample_rate,
        hop_length=hop_length
    )
    del onset_pred
    del frame_pred
    del velocity_pred

    p_ref, i_ref, v_ref = model.extract_notes(
        onset_true.squeeze(),
        frame_true.squeeze(),
        velocity_true.squeeze(),
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    del onset_true
    del frame_true
    del velocity_true

    sample_metrics.append(
        compute_note_metrics(i_est, p_est, v_est, i_ref, p_ref, v_ref)
    )

# average metrics over samples,
# noting that all metric dicts have the same structure
avg_metrics = defaultdict(dict)
for metric_type in sample_metrics[0].keys():
    for cls_metric in sample_metrics[0][metric_type]:
        avg_metrics[metric_type][cls_metric] = np.mean([
            metric[metric_type][cls_metric] for metric in sample_metrics
        ])

table = []
for metric_type in sorted(avg_metrics.keys()):
    d = avg_metrics[metric_type]
    table.append([
        metric_type,
        d["precision"],
        d["recall"],
        d["f1"],
        d["overlap"],
    ])

print(tabulate(table, headers=["precision", "recall", "f1", "overlap"]))
