import pytest
import torch

from virtual_stain_flow.trainers.trainer_utils.early_stop import (
    EarlyStopHelper,
    _get_latest_metric_value,
)


class TestGetLatestMetricValue:
    def test_gets_value_from_losses(self):
        value = _get_latest_metric_value(
            val_losses={"mse_loss": [0.5]},
            val_metrics={},
            metric_name="mse_loss",
        )

        assert value == 0.5

    def test_gets_value_from_metrics(self):
        value = _get_latest_metric_value(
            val_losses={},
            val_metrics={"accuracy": [0.85]},
            metric_name="accuracy",
        )

        assert value == 0.85

    def test_defaults_to_first_validation_loss(self):
        value = _get_latest_metric_value(
            val_losses={"mse_loss": [0.5], "l1_loss": [0.4]},
            val_metrics={},
        )

        assert value == 0.5

    def test_missing_metric_raises_error(self):
        with pytest.raises(ValueError, match="not found in logs"):
            _get_latest_metric_value({}, {}, "missing")

    def test_missing_default_loss_raises_error(self):
        with pytest.raises(RuntimeError, match="No validation losses"):
            _get_latest_metric_value({}, {})

    def test_empty_metric_history_raises_error(self):
        with pytest.raises(RuntimeError, match="has no recorded values"):
            _get_latest_metric_value({"mse_loss": []}, {}, "mse_loss")

    def test_invalid_metric_history_raises_error(self):
        with pytest.raises(TypeError, match="list or tuple"):
            _get_latest_metric_value({"mse_loss": 0.5}, {}, "mse_loss")


class TestEarlyStopHelper:
    def test_best_model_is_none_before_first_improvement(self):
        losses = {"val_loss": [0.5]}
        helper = EarlyStopHelper(
            model=torch.nn.Linear(1, 1),
            trainer_val_losses_ref=losses,
            best_metric_name="val_loss",
        )
        helper.initialize_early_stop(patience=2)

        assert helper.best_model is None

    def test_improvement_resets_counter(self):
        losses = {"val_loss": [0.5]}
        helper = EarlyStopHelper(
            model=torch.nn.Linear(1, 1),
            trainer_val_losses_ref=losses,
            best_metric_name="val_loss",
        )
        helper.initialize_early_stop(patience=3)
        assert helper.update() is False
        losses["val_loss"].append(0.6)
        assert helper.update() is False
        losses["val_loss"].append(0.4)

        should_stop = helper.update()

        assert should_stop is False
        assert helper.counter == 0
        assert helper.best_metric_value == 0.4

    def test_non_improvement_stops_at_patience(self):
        losses = {"val_loss": [0.3]}
        helper = EarlyStopHelper(
            model=torch.nn.Linear(1, 1),
            trainer_val_losses_ref=losses,
            best_metric_name="val_loss",
        )
        helper.initialize_early_stop(patience=2)
        assert helper.update() is False
        losses["val_loss"].append(0.4)
        assert helper.update() is False
        losses["val_loss"].append(0.5)

        assert helper.update() is True
        assert helper.counter == 2

    def test_disabled_update_is_no_op(self, minimal_model):
        helper = EarlyStopHelper(model=minimal_model, enabled=False)
        helper.initialize_early_stop(patience=1)

        should_stop = helper.update(epoch=1)

        assert should_stop is False
        assert helper.counter == 0
        assert helper.best_model is None

    def test_snapshot_tracks_internal_model_reference(self, minimal_model):
        losses = {"val_loss": [0.5]}
        helper = EarlyStopHelper(
            model=minimal_model,
            trainer_val_losses_ref=losses,
            best_metric_name="val_loss",
        )
        helper.initialize_early_stop(patience=2)

        with torch.no_grad():
            for parameter in minimal_model.parameters():
                parameter.add_(1.0)

        helper.update(epoch=1)

        assert helper.best_model is not None
        assert helper.best_model is not minimal_model
        for copied_parameter, parameter in zip(
            helper.best_model.parameters(), minimal_model.parameters()
        ):
            assert torch.equal(copied_parameter, parameter)

    def test_enabled_update_requires_initialization(self):
        helper = EarlyStopHelper(model=torch.nn.Linear(1, 1),trainer_val_losses_ref={"loss": [0.5]})

        with pytest.raises(RuntimeError, match="has not been initialized"):
            helper.update()

    def test_initialize_rejects_invalid_patience(self):
        helper = EarlyStopHelper(model=torch.nn.Linear(1, 1))

        with pytest.raises(ValueError, match="at least 1"):
            helper.initialize_early_stop(0)

    def test_reinitialize_resets_counter_and_preserves_best_state(self):
        losses = {"loss": [0.3]}
        helper = EarlyStopHelper(model=torch.nn.Linear(1, 1), trainer_val_losses_ref=losses)
        helper.initialize_early_stop(patience=2)
        helper.update(epoch=1)
        losses["loss"].append(0.4)
        helper.update(epoch=2)

        helper.initialize_early_stop(patience=3)

        assert helper.counter == 0
        assert helper.best_metric_value == 0.3
        assert helper.best_epoch_state.best_epoch == 1
