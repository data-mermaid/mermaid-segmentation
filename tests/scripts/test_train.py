"""Training entrypoint."""

from __future__ import annotations

import scripts.train as train_script


def test_main_passes_the_parsed_arguments_to_run(monkeypatch):
    argv = ["--num-workers", "8", "--cache-dir", "cache", "--resolution", "128"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv, "--init-weights", "w.pt"])
    received: list = []
    monkeypatch.setattr(train_script, "run", received.append)
    train_script.main()
    assert len(received) == 1
    assert received[0].num_workers == 8
    assert received[0].cache_dir == "cache"
    assert received[0].resolution == 128
    assert received[0].init_weights == "w.pt"
    assert received[0].checkpoint is None
