"""
Helper module for re-loading model from MLflow tracking info
"""

import pathlib
import re
import json
import importlib
from typing import Optional, Literal, Dict

import torch
from mlflow.tracking import MlflowClient
from mlflow.entities.file_info import FileInfo

from ..evaluation.as_gif import images_to_numbered_gif


def _artifact_sort_key(artifact_path: str):
    p = pathlib.Path(artifact_path)
    nums = [int(x) for x in re.findall(r"\d+", p.stem)]
    return (nums, p.stem)


def _get_weight_artifacts(
    client: MlflowClient,
    tracking_run_id: str
) -> list[FileInfo]:
    
    try:
        weight_artifacts = [
            item for item in client.list_artifacts(tracking_run_id, path="weights")
            if (not item.is_dir) and pathlib.Path(item.path).suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
        ]
    except Exception as e:
        raise ValueError(f"Failed to list artifacts for run ID '{tracking_run_id}': {e}")
    
    if not weight_artifacts:
        raise ValueError(f"No weight artifacts found for run ID '{tracking_run_id}' at path 'weights/'")

    return weight_artifacts


def _select_weight_artifact(
    client: MlflowClient, 
    tracking_run_id: str,
    load_weight_mode: Literal["latest", "best"] = "latest"
) -> FileInfo:
    
    weight_artifacts = _get_weight_artifacts(client, tracking_run_id)

    if load_weight_mode == "best":
        best_weight_artifact = [
            weight_file
            for weight_file in weight_artifacts
            if "best" in weight_file.path.lower()
        ]
        if not best_weight_artifact:
            raise ValueError(
                "No weight artifacts with 'best' in filename found "
                f"for run ID '{tracking_run_id}'")
        if len(best_weight_artifact) > 1:
            raise RuntimeError(
                "Multiple weight artifacts with 'best' in filename found "
                f"for run ID '{tracking_run_id}': {[a.path for a in best_weight_artifact]}")
        
        return best_weight_artifact[0]
    
    elif load_weight_mode == "latest":

        latest_weight_artifact = sorted(weight_artifacts, key=lambda x: _artifact_sort_key(x.path))[-1]

        return latest_weight_artifact
    

def _load_config(
    client: MlflowClient,
    tracking_run_id: str
) -> Dict:
    
    try:
        config_artifacts = client.list_artifacts(tracking_run_id, path="configs")
    except Exception as e:
        raise ValueError(f"Failed to list config artifacts for run ID '{tracking_run_id}': {e}")
    
    if not config_artifacts:
        raise ValueError(f"No config artifacts found for run ID '{tracking_run_id}' at path 'configs/'")

    generator_config_artifacts = [
        artifact for artifact in config_artifacts
        if (artifact.path.lower().endswith(".json") and all(keyword not in artifact.path.lower() for keyword in ["discriminator", "loss_group", "optimizer"]))
    ]
    if not generator_config_artifacts:
        raise ValueError(f"No generator config artifacts found for run ID '{tracking_run_id}'")
    if len(generator_config_artifacts) > 1:
        raise ValueError(f"Multiple generator config artifacts found for run ID '{tracking_run_id}': {[a.path for a in generator_config_artifacts]}")

    # TODO: need to extend support for multiple config files
    # Best place to start is probably not here but in the MLflow logging process
    # to ensure more consistent naming of config artifacts
    try:
        local_config_path = client.download_artifacts(tracking_run_id, generator_config_artifacts[0].path)
    except Exception as e:
        raise ValueError(f"Failed to download config artifact '{generator_config_artifacts[0].path}' for run ID '{tracking_run_id}': {e}")
    
    try:
        config = json.load(open(local_config_path, "r"))
    except Exception as e:
        raise ValueError(f"Failed to load config from '{local_config_path}': {e}")
    
    return config


def _artifact_path_exists(
    client: MlflowClient,
    tracking_run_id: str,
    artifact_path: str,
) -> bool:
    try:
        items = client.list_artifacts(tracking_run_id, path=artifact_path)
    except Exception:
        return False

    return len(items) > 0


# maybe this helper belong better somewhere else
def _get_class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as e:
        raise ValueError(f"Failed to get class from path '{class_path}': {e}")    
    
    return cls


def from_mlflow_run(
    tracking_run_id: str,
    tracking_uri: str,
    experiment_name: str,
    model_class: Optional[torch.nn.Module] = None,
    model_kwargs: Optional[Dict] = None,
    load_weight_mode: Literal["latest", "best"] = "latest",
    device: torch.device = 'cpu'
) -> torch.nn.Module:
    
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
    except Exception as e:
        raise ValueError(f"Failed to initialize MLflow client: {e}")
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found at URI '{tracking_uri}'")
    except Exception as e:
        raise ValueError(f"Failed to retrieve experiment '{experiment_name}': {e}")
    
    try:
        run = client.get_run(tracking_run_id)
        if run.info.experiment_id != experiment.experiment_id:
            raise ValueError(
                f"Run {tracking_run_id} belongs to experiment_id={run.info.experiment_id}, "
                f"not '{experiment_name}' (id={experiment.experiment_id})."
            )
    except Exception as e:
        raise ValueError(f"Failed to retrieve run with ID '{tracking_run_id}': {e}")
    
    selected_artifact = _select_weight_artifact(client, tracking_run_id, load_weight_mode)

    try:
        local_weight_path = client.download_artifacts(tracking_run_id, selected_artifact.path)
    except Exception as e:
        raise ValueError(f"Failed to download artifact '{selected_artifact.path}' for run ID '{tracking_run_id}': {e}")
    
    try:
        ckpt = torch.load(local_weight_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from '{local_weight_path}': {e}")
    
    if isinstance(ckpt, dict):
        if "generator_state_dict" in ckpt:
            state_dict = ckpt["generator_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # assume raw state_dict-like checkpoint
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    
    if model_class is not None and model_kwargs is not None:
        # case 1: user supplies both model class and kwargs
        # override config class path and kwargs if explicitly provided
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate model from class '{model_class}': {e} "
                f"with custom model specification {model_class} and {model_kwargs}"
            )
    elif model_class is not None and model_kwargs is None:
        # case 2: user supplies model class only
        # infer kwargs from config best effort and let it crash if needed
        # kwargs cannot be inferred
        config = _load_config(client, tracking_run_id)
        try:
            model = model_class(**config.get("init", {}))
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate model from class '{model_class}': {e} "
                f"with config specification {config.get('init', {})}"
            )
    elif model_class is None and model_kwargs is not None:
        # case 3: user supplies model kwargs without class
        # attempt to infer class from config but use override kwargs
        try:
            config = _load_config(client, tracking_run_id)
        except Exception as e:
            raise ValueError(
                f"Failed to load config for run ID '{tracking_run_id}': {e} "
                "for model class inferring"
            )
        
        try:
            model_class = _get_class_from_path(config['class_path'])
        except Exception as e:
            raise ValueError(
                f"Failed to get model class from config for run ID '{tracking_run_id}': {e} "
                "for model class inferring"
            )
        
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate model from class '{model_class}': {e} "
                f"with override kwargs {model_kwargs}"
            )
    else:
        # case 4: user does not supply model class or kwargs
        # attempt to infer both from config but let it crash if needed
        try:
            config = _load_config(client, tracking_run_id)
        except Exception as e:
            raise ValueError(
                f"Failed to load config for run ID '{tracking_run_id}': {e} "
                "for model class and kwargs inferring"
            )
        
        try:
            model_class = _get_class_from_path(config['class_path'])
        except Exception as e:
            raise ValueError(
                f"Failed to get model class from config for run ID '{tracking_run_id}': {e} "
                "for model class inferring"
            )
        
        try:
            model = model_class(**config.get("init", {}))
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate model from class '{model_class}': {e} "
                f"with config specification {config.get('init', {})}"
            )
        
    try:
        load_info = model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Failed to load state dict into model: {e}")
    
    return model, load_info


def gifs_from_mlflow_run(
    tracking_run_id: str,
    tracking_uri: str,
    output_dir: pathlib.Path,
    pattern: str = "epoch",
    fps: float = 5,
    number_color: str = "black",
    subset: Optional[list[int]] = None,
    font_size: int = 24,
    padding: int = 8,
    loop: int = 0,
) -> dict[str, pathlib.Path]:
    """
    Generate GIFs from common prediction plot folders in MLflow artifacts.
    """

    client = MlflowClient(tracking_uri=tracking_uri)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = [
        "plots/epoch/plot_train_predictions",
        "plots/epoch/plot_heldout_predictions",
    ]

    outputs: dict[str, pathlib.Path] = {}

    for artifact_path in artifact_paths:
        if not _artifact_path_exists(client, tracking_run_id, artifact_path):
            continue

        local_dir = pathlib.Path(
            client.download_artifacts(tracking_run_id, artifact_path)
        )

        output_path = output_dir / f"{tracking_run_id}_{local_dir.name}.gif"

        images_to_numbered_gif(
            image_dir=local_dir,
            output_path=output_path,
            pattern=pattern,
            fps=fps,
            number_color=number_color,
            subset=subset,
            font_size=font_size,
            padding=padding,
            loop=loop,
        )

        outputs[artifact_path] = output_path

    return outputs
