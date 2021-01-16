import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from PIL import Image


def make_midi_plots(
    onset_pred,
    frame_pred,
    onset_true,
    frame_true,
    hop_length=1,
    sample_rate=16000,
    onset_threshold=0.5,
    frame_threshold=0.5,
    note_range=(None, None),
    time_range=(None, None),
    scale_factor=4,
    figsize=None,
):
    # generating onset & notes plot
    onset_pred = onset_pred.squeeze().cpu().t().numpy() > onset_threshold
    frame_pred = frame_pred.squeeze().cpu().t().numpy() > frame_threshold

    accepted_frames = frame_pred.copy()
    # scanning the frames array to reject frames without an onset
    for t in range(frame_pred.shape[1]):
        prev = accepted_frames[:, t - 1] if t > 0 \
            else np.zeros(frame_pred.shape[0], dtype=np.bool)

        accepted_frames[:, t] = np.bitwise_or(
            np.bitwise_and(accepted_frames[:, t], onset_pred[:, t]),
            np.bitwise_and(accepted_frames[:, t], prev),
        )

    rejected_frames = np.bitwise_xor(frame_pred, accepted_frames)

    pred_all = np.bitwise_or(onset_pred, accepted_frames)

    img = np.ones((*onset_pred.shape, 3))
    img[accepted_frames] = (0.75, 0.00, 1.00)  # purple
    img[rejected_frames] = (0.00, 0.72, 0.92)  # cyan
    img[onset_pred] = (0.0, 0.0, 0.0)
    img = (img * 255).astype(np.uint8)
    img = img[note_range[0]:note_range[1], time_range[0]:time_range[1]]
    img = np.flip(img, axis=0)

    new_size = (s * scale_factor for s in img.shape[1::-1])
    img = Image.fromarray(img).resize(new_size, Image.NEAREST)
    img1 = np.array(img)

    # generating pred. vs. true frame_true plot
    onset_true = onset_true.squeeze().cpu().t().numpy().astype(np.bool)
    frame_true = frame_true.squeeze().cpu().t().numpy().astype(np.bool)

    true_all = np.bitwise_or(onset_true, frame_true)

    img = np.ones((*onset_true.shape, 3))
    img[true_all] = (1.0, 0.4, 0.0)  # orange
    img[pred_all] = (0.16, 0.55, 1.00)  # blue
    img[np.bitwise_and(true_all, pred_all)] = (0.85, 0.51, 1.00)  # purple
    img = (img * 255).astype(np.uint8)
    img = img[note_range[0]:note_range[1], time_range[0]:time_range[1]]
    img = np.flip(img, axis=0)

    new_size = (s * scale_factor for s in img.shape[1::-1])
    img = Image.fromarray(img).resize(new_size, Image.NEAREST)
    img2 = np.array(img)

    # plotting the images
    if figsize is None:
        figsize = (20, 10)
    plt.figure(figsize=figsize)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=200)

    axes[0].set_title("Estimated onsets and notes")
    axes[0].get_xaxis().set_visible(False)
    axes[0].set_ylabel("Note")
    axes[0].set_yticks([])
    axes[0].imshow(img1)

    axes[1].set_title("Estimated and reference transcription")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, pos: f"{x * hop_length / sample_rate / scale_factor}")
    )
    axes[1].set_ylabel("Note")
    axes[1].set_yticks([])
    axes[1].imshow(img2)
    plt.show()
