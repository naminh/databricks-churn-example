from churn.common import Workload, MLflowTrackingConfig
from churn import train
from churn.train import ModelTrain, ModelTrainConfig
from churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelTrainJob(Workload):

    def _get_mlflow_tracking_cfg(self) -> MLflowTrackingConfig:

        if self.env_vars['model_train_experiment_path']:
            return MLflowTrackingConfig(run_name=self.conf['mlflow_params']['run_name'],
                                        experiment_path=self.env_vars['model_train_experiment_path'],
                                        model_name=self.env_vars['model_name'])

        elif self.env_vars['model_train_experiment_id']:
            return MLflowTrackingConfig(run_name=self.conf['mlflow_params']['run_name'],
                                        experiment_id=self.env_vars['model_train_experiment_id'],
                                        model_name=self.env_vars['model_name'])
        else:
            raise RuntimeError('Either model_train_experiment_path or model_train_experiment_id must be passed as an '
                               'arg via the environment .env config')

    def _get_data_input_snowflake_cfg(self) -> train.DataInputSnowflakeConfig:

        return train.DataInputSnowflakeConfig(columns=self.conf['data_input']['snowflake_params']['columns'])

    def _get_data_input_feature_store_cfg(self) -> train.DataInputFeatureStoreConfig:

        feature_store_params = self.conf['data_input']['feature_store_params']

        # Use Database specified by env file, table names passed via config
        feature_table_list = [f'{self.env_vars["feature_store_database_name"]}.{tbl}' for tbl in feature_store_params['feature_table_list']]

        return train.DataInputFeatureStoreConfig(feature_table_list=feature_table_list,
                                                 primary_keys=feature_store_params['primary_keys'],
                                                 timestamp_keys=feature_store_params['timestamp_keys'])

    def launch(self):
        _logger.info('Launching ModelTrainJob job')

        if self.conf['data_input']['use_feature_store']:
            cfg = ModelTrainConfig(mlflow_tracking_cfg=self._get_mlflow_tracking_cfg(),
                                   data_input_snowflake_cfg=self._get_data_input_snowflake_cfg(),
                                   data_input_feature_store_cfg=self._get_data_input_feature_store_cfg(),
                                   pipeline_params=self.conf['pipeline_params'],
                                   model_params=self.conf['model_params'],
                                   conf=self.conf,
                                   env_vars=self.env_vars)
            ModelTrain(cfg).run(use_feature_store=True)

        else:
            cfg = ModelTrainConfig(mlflow_tracking_cfg=self._get_mlflow_tracking_cfg(),
                                   data_input_snowflake_cfg=self._get_data_input_snowflake_cfg(),
                                   pipeline_params=self.conf['pipeline_params'],
                                   model_params=self.conf['model_params'],
                                   conf=self.conf,
                                   env_vars=self.env_vars)
            ModelTrain(cfg).run(use_feature_store=False)

        _logger.info('ModelTrainJob job finished!')


if __name__ == '__main__':
    job = ModelTrainJob()
    job.launch()
