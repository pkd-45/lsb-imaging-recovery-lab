from pathlib import Path

from lsbrecovery.model import TinyUNet
from lsbrecovery.train import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "m.pt"
    save_checkpoint(TinyUNet(in_channels=3), path, mode="triplet")
    model, mode = load_checkpoint(path, "cpu")
    assert model.in_channels == 3
    assert model.out_channels == 2
    assert mode == "triplet"
