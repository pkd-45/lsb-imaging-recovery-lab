import numpy as np

from lsbrecovery.track import batch_stream_track_metrics, stream_track_metrics


def test_perfect_stream_track_is_usable():
    mask = np.zeros((64, 64), dtype=bool)
    cy = cx = 31.5
    yy, xx = np.indices(mask.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    mask[(rr > 18) & (rr < 21) & (theta > -1.0) & (theta < 1.0)] = True
    out = stream_track_metrics(mask, mask)
    assert out["track_completeness"] == 1.0
    assert out["track_radial_mae_fraction"] == 0.0
    assert out["usable"] == 1.0


def test_batch_track_uses_only_streams():
    mask = np.zeros((2, 32, 32), dtype=bool)
    mask[:, 10:12, 8:24] = True
    out = batch_stream_track_metrics(mask, mask, ["stream", "shell"])
    assert out["n_streams"] == 1.0
