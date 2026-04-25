"""
Tests for automatic loss-group config logging in MlflowLogger.
"""


class TestMlflowLoggerLossConfigLogging:

    def test_on_train_start_logs_single_trainer_loss_tags_and_config(
        self,
        patched_mlflow,
        single_generator_trainer,
    ):
        from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger

        captured = patched_mlflow

        logger = MlflowLogger(
            name='logger',
            experiment_name='exp',
        )
        logger.bind_trainer(single_generator_trainer)

        logger.on_train_start()

        assert captured['tags']['loss.main.0.name'] == 'MSELoss'
        assert captured['tags']['loss.main.0.weight'] == '1.0'

        loss_group_artifacts = [
            artifact
            for artifact in captured['artifacts']
            if artifact['content'] is not None
            and artifact['content'].get('group_name') == 'main'
        ]

        assert len(loss_group_artifacts) == 1
        artifact = loss_group_artifacts[0]
        assert artifact['artifact_path'] == 'configs'
        assert len(artifact['content']['items']) == 1
        assert artifact['content']['items'][0]['key'] == 'MSELoss'
        assert artifact['content']['items'][0]['weight'] == 1.0

        logger.end_run()

    def test_on_train_start_logs_wgan_loss_tags_and_configs(
        self,
        patched_mlflow,
        wgan_trainer,
    ):
        from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger

        captured = patched_mlflow

        logger = MlflowLogger(
            name='logger',
            experiment_name='exp',
        )
        logger.bind_trainer(wgan_trainer)

        logger.on_train_start()

        assert captured['tags']['loss.generator.0.name'] == 'MSELoss'
        assert captured['tags']['loss.generator.0.weight'] == '1.0'
        assert captured['tags']['loss.generator.1.name'] == 'AdversarialLoss'
        assert captured['tags']['loss.generator.1.weight'] == '1.0'

        assert captured['tags']['loss.discriminator.0.name'] == 'WassersteinLoss'
        assert captured['tags']['loss.discriminator.0.weight'] == '1.0'
        assert captured['tags']['loss.discriminator.1.name'] == 'GradientPenaltyLoss'
        assert captured['tags']['loss.discriminator.1.weight'] == '10.0'

        group_names = {
            artifact['content']['group_name']
            for artifact in captured['artifacts']
            if artifact['content'] is not None
            and 'group_name' in artifact['content']
        }

        assert group_names == {'generator', 'discriminator'}

        logger.end_run()
