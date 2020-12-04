from collections import defaultdict

from mir_eval.transcription import precision_recall_f1_overlap as eval_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as eval_notes_with_velocity,
)
from sklearn.metrics import precision_score, recall_score


def compute_note_metrics(i_est, p_est, v_est, i_ref, p_ref, v_ref):
    metrics = defaultdict(dict)

    p, r, f1, o = eval_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics["note"]["precision"] = p
    metrics["note"]["recall"] = r
    metrics["note"]["f1"] = f1
    metrics["note"]["overlap"] = o

    p, r, f1, o = eval_notes(i_ref, p_ref, i_est, p_est)
    metrics["note_with_offsets"]["precision"] = p
    metrics["note_with_offsets"]["recall"] = r
    metrics["note_with_offsets"]["f1"] = f1
    metrics["note_with_offsets"]["overlap"] = o

    p, r, f1, o = eval_notes_with_velocity(
        i_ref,
        p_ref,
        v_ref,
        i_est,
        p_est,
        v_est,
        offset_ratio=None,
        velocity_tolerance=0.1,
    )
    metrics["note_with_velocity"]["precision"] = p
    metrics["note_with_velocity"]["recall"] = r
    metrics["note_with_velocity"]["f1"] = f1
    metrics["note_with_velocity"]["overlap"] = o

    p, r, f1, o = eval_notes_with_velocity(
        i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1
    )
    metrics["note_with_offsets_and_velocity"]["precision"] = p
    metrics["note_with_offsets_and_velocity"]["recall"] = r
    metrics["note_with_offsets_and_velocity"]["f1"] = f1
    metrics["note_with_offsets_and_velocity"]["overlap"] = o

    return metrics


def compute_frame_metrics(frame_pred, frame_true):
    p = precision_score(frame_true.T, frame_pred.T, average="micro")
    r = recall_score(frame_true.T, frame_pred.T, average="micro")
    f1 = (2 * p * r) / (p + r)

    return {"frame": {
        "precision": p,
        "recall": r,
        "f1": f1
    }}
