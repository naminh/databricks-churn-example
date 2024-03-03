from dataclasses import dataclass
from typing import List, Dict, Any, Union
import pprint

import pandas as pd
import pyspark.sql.dataframe
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pyspark.sql.functions as f

import databricks
from databricks.feature_store import FeatureStoreClient, FeatureLookup

from churn.common import MLflowTrackingConfig
from churn.model import ModelPipeline
from churn.utils.get_spark import spark
from churn.utils.logger_utils import get_logger

fs = FeatureStoreClient()
client = MlflowClient()
_logger = get_logger()


@dataclass
class DataInputSnowflakeConfig:
    columns: List[str]


@dataclass
class DataInputFeatureStoreConfig:
    feature_table_list: List[str]
    primary_keys: List[str]
    timestamp_keys: List[str]


@dataclass
class ModelTrainConfig:
    """
    Configuration data class used to execute ModelTrain pipeline.
    Attributes:
        mlflow_tracking_cfg (MLflowTrackingConfig)
            Configuration data class used to unpack MLflow parameters during a model training run.
        data_input_cfg (ModelTrainDataInputConfig):
            Configuration data class used to unpack parameters when loading data from Snowflake and Feature Store tables.
        pipeline_params (dict):
            Params to use in preprocessing pipeline. Read from model_train.yml
            - test_size: Proportion of input data to use as training data
            - random_state: Random state to enable reproducible train-test split
        model_params (dict):
            Dictionary of params for model. Read from model_train.yml
        conf (dict):
            [Optional] dictionary of conf file used to trigger pipeline. If provided will be tracked as a yml
            file to MLflow tracking.
        env_vars (dict):
            [Optional] dictionary of environment variables to trigger pipeline. If provided will be tracked as a yml
            file to MLflow tracking.
    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    pipeline_params: Dict[str, Any]
    model_params: Dict[str, Any]
    data_input_snowflake_cfg: DataInputSnowflakeConfig
    data_input_feature_store_cfg: DataInputFeatureStoreConfig = None
    conf: Dict[str, Any] = None
    env_vars: Dict[str, str] = None


class ModelTrain:
    """
    Class to execute model training. Params, metrics and model artifacts will be tracking to MLflow Tracking.
    Optionally, the resulting model will be registered to MLflow Model Registry if provided.
    """

    def __init__(self, cfg: ModelTrainConfig):
        self.cfg = cfg

    @staticmethod
    def _set_experiment(mlflow_tracking_cfg: MLflowTrackingConfig):
        """
        Set MLflow experiment. Use one of either experiment_id or experiment_path
        """
        if mlflow_tracking_cfg.experiment_id is not None:
            _logger.info(f'MLflow experiment_id: {mlflow_tracking_cfg.experiment_id}')
            mlflow.set_experiment(experiment_id=mlflow_tracking_cfg.experiment_id)
        elif mlflow_tracking_cfg.experiment_path is not None:
            _logger.info(f'MLflow experiment_path: {mlflow_tracking_cfg.experiment_path}')
            mlflow.set_experiment(experiment_name=mlflow_tracking_cfg.experiment_path)
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')

    def _get_snowflake_features_and_label_df(self) -> pyspark.sql.DataFrame:
        """
        Load DataFrame from Snowflake using hardcoded query in churn.transform module

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing columns defined in the config file, along with the label col
        """

        pass

    @staticmethod
    def _get_feature_table_lookup(feature_table_name: str,
                                  primary_keys: Union[str, List[str]],
                                  timestamp_lookup_key: Union[str, List[
                                      str]]) -> databricks.feature_store.entities.feature_lookup.FeatureLookup:
        """
        Create list of FeatureLookup for single feature store table. The FeatureLookup is a value class used to specify
        features to use in a TrainingSet.

        Parameters
        ----------
        feature_table_name :  str
            Feature table name
        primary_keys : Union[str, List[str]]
            Key(s) to use when joining this feature table with the DataFrame passed to FeatureStoreClient.create_training_set().
            The lookup_key must be the columns in the DataFrame passed to FeatureStoreClient.create_training_set().
            The type of lookup_key columns in that DataFrame must match the type of the primary key of the feature table
            referenced in this FeatureLookup.
        timestamp_lookup_key :
            Key to use when performing point-in-time lookup on this feature table with the DataFrame passed to
            FeatureStoreClient.create_training_set().
            The timestamp_lookup_key must be the columns in the DataFrame passed to
            FeatureStoreClient.create_training_set().
            The type of timestamp_lookup_key columns in that DataFrame must match the type of the timestamp key of the
            feature table referenced in this FeatureLookup.

        Returns
        -------
        List[databricks.feature_store.entities.feature_lookup.FeatureLookup]
        """
        pass

    def get_fs_training_set(self,
                            df: pyspark.sql.DataFrame) -> databricks.feature_store.training_set.TrainingSet:
        """
        Create the Feature Store TrainingSet

        Returns
        -------
        databricks.feature_store.training_set.TrainingSet
        """

        pass

    def create_train_test_split(self, pdf: pd.DataFrame):
        """
        Load the TrainingSet for training. The loaded DataFrame has columns specified by fs_training_set.
        Loaded Spark DataFrame is converted to pandas DataFrame and split into train/test splits.

        Parameters
        ----------
        pdf : pd.DataFrame
            Pandas DataFrame containing features and label col
        Returns
        -------
        train-test splits
        """
        X = pdf.drop(self.cfg.pipeline_params['label_col'], axis=1)
        y = pdf[self.cfg.pipeline_params['label_col']]

        _logger.info(f'Splitting into train/test splits - test_size: {self.cfg.pipeline_params["test_size"]}')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=self.cfg.pipeline_params['random_state'],
                                                            test_size=self.cfg.pipeline_params['test_size'],
                                                            stratify=y)

        return X_train, X_test, y_train, y_test

    def fit_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.pipeline.Pipeline:
        """
        Create sklearn pipeline and fit pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data
        y_train : pd.Series
            Training labels
        Returns
        -------
        scikit-learn pipeline with fitted steps.
        """
        _logger.info('Creating sklearn pipeline...')
        pipeline = ModelPipeline.create_baseline_pipeline(self.cfg.model_params)

        _logger.info('Fitting sklearn RandomForestClassifier...')
        _logger.info(f'Model params: {pprint.pformat(self.cfg.model_params)}')
        model = pipeline.fit(X_train, y_train)

        return model

    def run(self, use_feature_store: bool = False):
        """
        Method to trigger model training, and tracking to MLflow.
        Steps:
            1. Set MLflow experiment (creating a new experiment if it does not already exist)
            2. Start MLflow run
            3. Load features and label col from Snowflake
            4. If use_feature_store=True, create Databricks Feature Store training set
            5. Create train-test splits to be used to train and evaluate the model
            6. Define sklearn pipeline using ModelTrainPipeline, and fit on train data
            7. If use_feature_store=True, log trained model using the Databricks Feature Store API. Model will be logged
               to MLflow with associated feature table metadata.
               Note that when not using the Feature Store API, the standard MLflow model is automatically logged.
            8. Register the model to MLflow model registry if model_name is provided in mlflow_params
        """
        _logger.info('==========Running model training==========')
        _logger.info(f'use_feature_store={use_feature_store}')
        mlflow_tracking_cfg = self.cfg.mlflow_tracking_cfg

        _logger.info('==========Setting MLflow experiment==========')
        self._set_experiment(mlflow_tracking_cfg)
        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        _logger.info('==========Starting MLflow run==========')
        with mlflow.start_run(run_name=mlflow_tracking_cfg.run_name) as mlflow_run:

            if self.cfg.conf is not None:
                # Log config file
                mlflow.log_dict(self.cfg.conf, 'conf.yml')
            if self.cfg.env_vars is not None:
                # Log env_vars file
                mlflow.log_dict(self.cfg.env_vars, 'env_vars.yml')

            # Load features from Snowflake
            _logger.info('==========Loading features from Snowflake==========')
            snowflake_df = pd.DataFrame() # TODO: removed actual method

            # Only use Snowflake features
            input_df = snowflake_df
            input_pdf = input_df.toPandas()

            # Load and preprocess data into train/test splits
            _logger.info('==========Creating train/test splits==========')
            X_train, X_test, y_train, y_test = self.create_train_test_split(input_pdf)
            mlflow.log_param('train_count', X_train.shape[0])
            mlflow.log_param('test_count', X_test.shape[0])

            # Fit pipeline with RandomForestClassifier
            _logger.info('==========Fitting RandomForestClassifier model==========')
            model = self.fit_pipeline(X_train, y_train)

            # Training metrics are logged by MLflow autologging
            # Log metrics for the test set
            _logger.info('==========Model Evaluation==========')
            _logger.info('Evaluating and logging metrics')
            test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix='test_')
            print(pd.DataFrame(test_metrics, index=[0]))

            # Register model to MLflow Model Registry if provided and register to stage='Staging'
            if mlflow_tracking_cfg.model_name:
                _logger.info('==========MLflow Model Registry==========')
                _logger.info(f'Registering model: {mlflow_tracking_cfg.model_name}')
              
                model_version = mlflow.register_model(f'runs:/{mlflow_run.info.run_id}/model',
                                                        name=mlflow_tracking_cfg.model_name)

                _logger.info(f'Transitioning model version {model_version.version} to stage="Staging"')
                client.transition_model_version_stage(name=mlflow_tracking_cfg.model_name,
                                                      version=model_version.version,
                                                      stage='Staging',
                                                      archive_existing_versions=True)

        _logger.info('==========Model training completed==========')
