from types import SimpleNamespace

import pytest
import torch

from virtual_stain_flow.models.model import BaseModel
from virtual_stain_flow.vsf_logging.auto_loggers.model_config_logger import (
	AutoModelConfigLogger,
)
from virtual_stain_flow.vsf_logging.auto_loggers.optimizer_config_logger import (
	AutoOptimizerConfigLogger,
)
from virtual_stain_flow.vsf_logging.auto_loggers.loss_group_config_logger import (
	AutoLossGroupConfigLogger,
)


class _DummyLogger:
	def __init__(self):
		self.logged = []

	def log_config(self, tag, config, stage=None):
		self.logged.append(
			{
				"tag": tag,
				"config": config,
				"stage": stage,
			}
		)


class _FailingLogger(_DummyLogger):
	def log_config(self, tag, config, stage=None):
		raise RuntimeError("forced log failure")


def _make_optimizer():
	model = torch.nn.Linear(4, 2)
	return torch.optim.Adam(model.parameters(), lr=1e-3)


class _FakeLossGroup:
	def __init__(self, config):
		self._config = config

	def get_config(self):
		return self._config


class _FakeModel(BaseModel):
	def __init__(self, config):
		super().__init__()
		self._config = config

	def forward(self, x):
		return x

	def to_config(self):
		return self._config

	@classmethod
	def from_config(cls, config):
		return cls(config)


def test_discover_optimizers_supports_list_and_single():
	logger = _DummyLogger()
	auto_logger = AutoOptimizerConfigLogger(logger)

	opt_a = _make_optimizer()
	opt_b = _make_optimizer()
	trainer = SimpleNamespace(optimizers=[opt_a], optimizer=opt_b)

	optimizers = auto_logger.discover_optimizers(trainer)

	assert optimizers == [opt_a, opt_b]


def test_discover_optimizers_returns_empty_for_none_trainer():
	logger = _DummyLogger()
	auto_logger = AutoOptimizerConfigLogger(logger)

	assert auto_logger.discover_optimizers(None) == []


def test_log_optimizer_configs_sets_class_path_tags_and_artifacts(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoOptimizerConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.optimizer_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	optimizer = _make_optimizer()
	trainer = SimpleNamespace(optimizer=optimizer)

	auto_logger.log_optimizer_configs(trainer)

	assert "optimizer.0.class_path" in captured_tags
	assert captured_tags["optimizer.0.class_path"].endswith("Adam")

	assert len(logger.logged) == 1
	assert logger.logged[0]["tag"] == "optimizer_0"
	assert logger.logged[0]["config"]["class_path"].endswith("Adam")
	assert logger.logged[0]["config"]["defaults"]["lr"] == pytest.approx(1e-3)


def test_log_optimizer_configs_skips_non_optimizer_entries(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoOptimizerConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.optimizer_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	trainer = SimpleNamespace(optimizers=["not-an-optimizer"])

	auto_logger.log_optimizer_configs(trainer)

	assert captured_tags == {}
	assert logger.logged == []


def test_log_optimizer_configs_swallows_log_config_failures(monkeypatch):
	logger = _FailingLogger()
	auto_logger = AutoOptimizerConfigLogger(logger)

	def fake_set_tag(_key, _value):
		return None

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.optimizer_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	trainer = SimpleNamespace(optimizer=_make_optimizer())

	# Should not raise despite logger.log_config raising.
	auto_logger.log_optimizer_configs(trainer)


def test_discover_models_prefers_models_list_over_single_model():
	logger = _DummyLogger()
	auto_logger = AutoModelConfigLogger(logger)

	model_a = _FakeModel({"class_path": "pkg.ModelA", "init": {}})
	model_b = _FakeModel({"class_path": "pkg.ModelB", "init": {}})
	trainer = SimpleNamespace(_models=[model_a], model=model_b)

	models = auto_logger._discover_models(trainer)

	assert models == [model_a]


def test_log_model_configs_sets_class_path_tag_and_artifact(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoModelConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.model_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	model = _FakeModel({"class_path": "virtual_stain_flow.models.unet.UNet", "init": {"depth": 4}})
	trainer = SimpleNamespace(model=model)

	auto_logger.log_model_configs(trainer)

	assert captured_tags["model.0.class_path"].endswith("UNet")
	assert len(logger.logged) == 1
	assert logger.logged[0]["tag"] == "_FakeModel"
	assert logger.logged[0]["config"]["init"]["depth"] == 4


def test_log_model_configs_skips_non_dict_configs(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoModelConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.model_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	model = _FakeModel(["not", "a", "dict"])
	trainer = SimpleNamespace(model=model)

	auto_logger.log_model_configs(trainer)

	assert captured_tags == {}
	assert logger.logged == []


def test_log_model_configs_swallows_log_config_failures(monkeypatch):
	logger = _FailingLogger()
	auto_logger = AutoModelConfigLogger(logger)

	def fake_set_tag(_key, _value):
		return None

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.model_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	model = _FakeModel({"class_path": "pkg.Model", "init": {}})
	trainer = SimpleNamespace(model=model)

	# Should not raise despite logger.log_config raising.
	auto_logger.log_model_configs(trainer)


def test_discover_loss_groups_supports_explicit_and_fallback_attrs():
	logger = _DummyLogger()
	auto_logger = AutoLossGroupConfigLogger(logger)

	main_group = _FakeLossGroup([{"key": "MSELoss", "weight": 1.0}])
	gen_group = _FakeLossGroup([{"key": "L1Loss", "weight": 0.5}])
	trainer = SimpleNamespace(
		loss_groups={"main": main_group},
		_generator_loss_group=gen_group,
	)

	loss_groups = auto_logger.discover_loss_groups(trainer)

	assert set(loss_groups.keys()) == {"main", "generator"}
	assert loss_groups["main"] is main_group
	assert loss_groups["generator"] is gen_group


def test_log_loss_group_configs_sets_tags_and_logs_config_artifact(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoLossGroupConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.loss_group_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	group_items = [
		{"key": "MSELoss", "weight": 1.0},
		{"key": "L1Loss", "weight": 0.25},
		{"key": None, "weight": None},
		"ignored-non-dict-item",
	]
	trainer = SimpleNamespace(loss_groups={"main": _FakeLossGroup(group_items)})

	auto_logger.log_loss_group_configs(trainer)

	assert captured_tags["loss.main.0.name"] == "MSELoss"
	assert captured_tags["loss.main.0.weight"] == "1.0"
	assert captured_tags["loss.main.1.name"] == "L1Loss"
	assert captured_tags["loss.main.1.weight"] == "0.25"

	assert len(logger.logged) == 1
	assert logger.logged[0]["tag"] == "loss_group_main"
	assert logger.logged[0]["config"]["group_name"] == "main"
	assert logger.logged[0]["config"]["items"] == group_items


def test_log_loss_group_configs_skips_non_list_config(monkeypatch):
	logger = _DummyLogger()
	auto_logger = AutoLossGroupConfigLogger(logger)

	captured_tags = {}

	def fake_set_tag(key, value):
		captured_tags[key] = value

	monkeypatch.setattr(
		"virtual_stain_flow.vsf_logging.auto_loggers.loss_group_config_logger.mlflow.set_tag",
		fake_set_tag,
	)

	trainer = SimpleNamespace(loss_groups={"main": _FakeLossGroup({"not": "a-list"})})

	auto_logger.log_loss_group_configs(trainer)

	assert captured_tags == {}
	assert logger.logged == []
