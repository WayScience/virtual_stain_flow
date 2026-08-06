from typing import Optional, List
import pathlib
import tempfile

import mlflow

from virtual_stain_flow.trainers.trainer_protocol import TrainerProtocol


def _log_artifact(
    file_path: pathlib.Path,
    artifact_path: Optional[str] = None,
    artifact_subdirs: Optional[List[str]] = None
) -> None:
    """
    Logs a single artifact to MLflow.

    :param file_path: The path to the file to log as an artifact.
    :param artifact_path: Optional artifact path within the MLflow run, defaults to None.
    :param artifact_subdirs: Optional list of subdirectories to include in the artifact path, defaults to None.
    :raises TypeError: If file_path is not a pathlib.Path instance.
    """

    if not isinstance(file_path, pathlib.Path):
        raise TypeError("file_path must be a pathlib.Path instance.")

    if artifact_path is None:
        artifact_ext = file_path.suffix.lower()
        if artifact_ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
            # log as plot artifact
            artifact_path = 'plots'
        elif artifact_ext in ['.pth', '.pt']:
            # log as model artifact
            artifact_path = 'weights'
        else:
            # log as generic artifact
            artifact_path = 'artifacts'

    artifact_subdirs = [] if artifact_subdirs is None else artifact_subdirs

    path_parts = [artifact_path, *artifact_subdirs]
    clean_parts = [
        part.replace('\\', '/').strip('/')
        for part in path_parts
        if part and part.strip('/\\')
    ]

    # Build a lexical POSIX path for MLflow without resolving it on the host OS.
    artifact_path = str(pathlib.PurePosixPath(*clean_parts)) if clean_parts else ''

    mlflow.log_artifact(str(file_path), artifact_path=artifact_path)


def _log_trainer_artifact(
    trainer: "TrainerProtocol",
    best_model: bool = True,
) -> None:
    """
    Logs the model and optimizer state artifacts from a trainer to MLflow.
    The most recent optimizer state is saved and logged regardless of the best_model flag.

    :param trainer: The trainer instance adhering to TrainerProtocol.
    :param best_model: Whether to log only the best model, defaults to True.
    :raises TypeError: If the provided trainer does not adhere to TrainerProtocol.
    """

    if not isinstance(trainer, TrainerProtocol):
        raise TypeError("The provided trainer must adhere to the TrainerProtocol.")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        tmpdirpath = pathlib.Path(tmpdirname)
        saved_model_paths = trainer.save_model(
            save_path=tmpdirpath, best_model=best_model
        )
        for saved_model_path in (saved_model_paths or []):
            _log_artifact(saved_model_path, artifact_path='weights')

        saved_optimizer_paths = trainer.save_optimizer_state(
            save_path=tmpdirpath, recent=True
        )
        for saved_optimizer_path in (saved_optimizer_paths or []):
            _log_artifact(saved_optimizer_path, artifact_path='optimizer')
