from argparse import ArgumentParser
from collections import defaultdict

import torch
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from src.data.audio import onf_transform
from src.data.data_modules import MAPSDataModule
from src.eval import compute_note_metrics, compute_frame_metrics
from src.models.onsets_and_frames import OnsetsAndFrames


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# testing over full audio files
dm = MAPSDataModule(batch_size=1,
                    sample_rate=16000,
                    max_steps=None,
                    audio_transform=onf_transform,
                    hop_length=512,
                    lazy_loading=True,
                    debug=args.debug)
dm.setup(stage="test")

model = OnsetsAndFrames.load_from_checkpoint(args.checkpoint,
                                             in_feats=229)
model.to(device).eval()

# a bit different from the validation loop, which partially works with batches
sample_metrics = []  # each elem is a metric dict for 1 sample
for batch in tqdm(dm.test_dataloader(), "Test samples"):
    # unpack data from batch (of 1 sample actually)
    audio_feats = batch["audio"].to(device)
    onset_true, frame_true, velocity_true = (
        batch["onsets"].to(device),
        batch["frames"].to(device),
        batch["velocity"].to(device),
    )
    sample_rate = batch["sample_rate"][0]
    hop_length = batch["hop_length"][0]

    with torch.no_grad():
        onset_pred, frame_pred, velocity_pred = model(audio_feats)

    p_est, i_est, v_est, final_frame_pred = model.extract_notes(
        onset_pred.squeeze(),
        frame_pred.squeeze(),
        velocity_pred.squeeze(),
        sample_rate=sample_rate,
        hop_length=hop_length
    )
    p_ref, i_ref, v_ref = batch["midi_labels"][0]

    sample_metrics.append({
        **compute_frame_metrics(final_frame_pred,
                                frame_true.squeeze().cpu().numpy()),
        **compute_note_metrics(i_est, p_est, v_est, i_ref, p_ref, v_ref)
    })

# average metrics over samples,
# noting that all metric dicts have the same structure
final_metrics = defaultdict(lambda: defaultdict(lambda: "N/A"))
for metric_type in sample_metrics[0].keys():
    for cls_metric in sample_metrics[0][metric_type]:
        vals = [metric[metric_type][cls_metric] for metric in sample_metrics]
        final_metrics[metric_type][cls_metric] = f"{np.mean(vals):.4f} \u00B1 {np.std(vals):.4f}"

table = []
for metric_type in sorted(final_metrics.keys()):
    d = final_metrics[metric_type]
    table.append([
        metric_type,
        d["precision"],
        d["recall"],
        d["f1"],
        d["overlap"],
    ])

print(tabulate(table, headers=["precision", "recall", "f1", "overlap"]))
