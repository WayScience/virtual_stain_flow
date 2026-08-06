"""Standalone tests for save_model helper."""

from types import SimpleNamespace

from virtual_stain_flow.trainers.trainer_utils.save_model import save_model


def test_save_model_returns_empty_when_target_model_is_none(tmp_path):
    trainer = SimpleNamespace(model=None, best_model=None, epoch=2)

    saved_paths = save_model(trainer=trainer, save_path=tmp_path, save_best_model=True)

    assert saved_paths == []


def test_save_model_saves_current_model_with_default_name(mock_model_with_save, tmp_path):
    trainer = SimpleNamespace(model=mock_model_with_save, best_model=None, epoch=3)

    saved_paths = save_model(trainer=trainer, save_path=tmp_path, save_best_model=False)

    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].name == "generator_weights_3.pth"
