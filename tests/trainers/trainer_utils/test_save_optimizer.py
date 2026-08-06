"""Standalone tests for save_optimizer_state helper."""

from types import SimpleNamespace

from virtual_stain_flow.trainers.trainer_utils.save_optimizer import save_optimizer_state


def test_save_optimizer_state_returns_empty_for_missing_optimizer(tmp_path):
    trainer = SimpleNamespace(optimizer=None, epoch=7)

    saved_paths = save_optimizer_state(trainer=trainer, save_path=tmp_path)

    assert saved_paths == []


def test_save_optimizer_state_saves_with_default_name(minimal_optimizer, tmp_path):
    trainer = SimpleNamespace(optimizer=minimal_optimizer, epoch=7)

    saved_paths = save_optimizer_state(trainer=trainer, save_path=tmp_path)

    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].name == "optimizer_7.pth"
