import unittest
import pandas as pd
import torch
import os
import tempfile
import shutil
import uuid
from unittest.mock import Mock, patch
from ml_utils.log_util import Logger, LoggerArgs, ExperimentMetrics  # Update import to match your file

class TestLoggerBasics(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.logger = Logger(experiment_name="test_experiment")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization_default_values(self):
        """Test default initialization values"""
        logger = Logger()
        self.assertEqual(logger.log_interval, 1)
        self.assertEqual(logger.wandb_interval, 50)
        self.assertEqual(logger.wandb_watch_interval, 1000)
        self.assertEqual(logger.experiment_name, "experiment")
        self.assertIsNone(logger.wandb_run)
        self.assertIsInstance(logger.run_id, str)

    def test_initialization_custom_values(self):
        """Test initialization with custom values"""
        custom_logger = Logger(
            log_interval=5,
            wandb_interval=100,
            wandb_watch_interval=2000,
            experiment_name="custom_experiment",
            run_id="custom_id"
        )
        self.assertEqual(custom_logger.log_interval, 5)
        self.assertEqual(custom_logger.wandb_interval, 100)
        self.assertEqual(custom_logger.wandb_watch_interval, 2000)
        self.assertEqual(custom_logger.experiment_name, "custom_experiment")
        self.assertEqual(custom_logger.run_id, "custom_id")

    def test_initialization_with_empty_values(self):
        """Test initialization with empty or None values"""
        logger = Logger(experiment_name="")
        self.assertEqual(logger.experiment_name, "")
        self.assertIsInstance(logger.run_id, str)

class TestLoggerMetricsHandling(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(experiment_name="test_experiment")

    def test_log_single_metric(self):
        """Test logging a single metric"""
        self.logger.log({"loss": 0.5})
        self.assertEqual(len(self.logger.metrics), 1)
        self.assertEqual(self.logger.metrics[0]["loss"], 0.5)

    def test_log_multiple_metrics(self):
        """Test logging multiple metrics at once"""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89
        }
        self.logger.log(metrics)
        self.assertEqual(self.logger.metrics[0], metrics)

    def test_log_nested_metrics(self):
        """Test logging nested metrics structure"""
        nested_metrics = {
            "training": {"loss": 0.5, "accuracy": 0.95},
            "validation": {"loss": 0.6, "accuracy": 0.93}
        }
        self.logger.log(nested_metrics)
        self.assertEqual(self.logger.metrics[0], nested_metrics)

    def test_log_different_numeric_types(self):
        """Test logging different numeric types"""
        metrics = {
            "float": 0.5,
            "int": 42,
            "scientific": 1.23e-4,
            "tensor_scalar": torch.tensor(0.7),
            "tensor_array": torch.tensor([0.8, 0.9]).mean(),
        }
        self.logger.log(metrics)
        for value in self.logger.metrics[0].values():
            self.assertIsInstance(value, (int, float))

    def test_log_at_intervals(self):
        """Test logging at specific intervals"""
        logger = Logger(log_interval=2)
        for i in range(5):
            logger.log({"step": i})
        # Should only log at steps 0, 2, 4
        self.assertEqual(len(logger.metrics), 3)
        self.assertEqual([m["step"] for m in logger.metrics], [0, 2, 4])

    def test_log_same_iter_updates(self):
        """Test updating metrics within the same iteration"""
        self.logger.log({"loss": 0.5})
        self.logger.log_same_iter({"accuracy": 0.95})
        self.logger.log_same_iter({"precision": 0.92})
        
        self.assertEqual(len(self.logger.metrics), 1)
        expected = {"loss": 0.5, "accuracy": 0.95, "precision": 0.92}
        self.assertEqual(self.logger.metrics[0], expected)

class TestLoggerDataFrameOperations(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()

    def test_empty_dataframe(self):
        """Test DataFrame conversion with no metrics"""
        df = self.logger.to_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_dataframe_with_metrics(self):
        """Test DataFrame conversion with various metrics"""
        metrics_list = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.4, "accuracy": 0.92},
            {"loss": 0.3, "accuracy": 0.94}
        ]
        for metrics in metrics_list:
            self.logger.log(metrics)
        
        df = self.logger.to_df()
        self.assertEqual(len(df), 3)
        self.assertTrue(all(col in df.columns for col in ["loss", "accuracy"]))
        self.assertTrue(df["loss"].is_monotonic_decreasing)
        self.assertTrue(df["accuracy"].is_monotonic_increasing)

    def test_dataframe_with_missing_values(self):
        """Test DataFrame handling of missing values"""
        self.logger.log({"loss": 0.5, "accuracy": 0.9})
        self.logger.log({"loss": 0.4})  # Missing accuracy
        self.logger.log({"accuracy": 0.94})  # Missing loss
        
        df = self.logger.to_df()
        self.assertEqual(len(df), 3)
        self.assertTrue(df["loss"].isna().any())
        self.assertTrue(df["accuracy"].isna().any())

class TestLoggerFileOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.logger = Logger(experiment_name="test_experiment")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_empty_experiment(self):
        """Test saving experiment with no metrics"""
        self.logger.save_experiment(self.test_dir)
        summary_path = os.path.join(self.test_dir, "test_experiment", "runs_summary.csv")
        self.assertTrue(os.path.exists(summary_path))

    def test_save_experiment_with_metrics(self):
        """Test saving experiment with metrics"""
        self.logger.hyper_params_dict = {"learning_rate": 0.01}
        self.logger.log({"loss": 0.5, "accuracy": 0.9})
        self.logger.save_experiment(self.test_dir)
        
        # Check files exist
        summary_path = os.path.join(self.test_dir, "test_experiment", "runs_summary.csv")
        run_path = os.path.join(self.test_dir, "test_experiment", f"run_{self.logger.run_id}.csv")
        
        self.assertTrue(os.path.exists(summary_path))
        self.assertTrue(os.path.exists(run_path))
        
        # Verify content
        summary_df = pd.read_csv(summary_path)
        run_df = pd.read_csv(run_path)
        
        self.assertEqual(summary_df["learning_rate"].iloc[0], 0.01)
        self.assertEqual(run_df["loss"].iloc[0], 0.5)
        self.assertEqual(run_df["accuracy"].iloc[0], 0.9)

    def test_save_experiment_directory_creation(self):
        """Test directory creation during save"""
        nested_dir = os.path.join(self.test_dir, "nested", "path")
        self.logger.save_experiment(nested_dir)
        self.assertTrue(os.path.exists(os.path.join(nested_dir, "test_experiment")))
import unittest
import pandas as pd
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from dataclasses import dataclass

@dataclass
class MockHyperParams:
    learning_rate: float = 0.01
    batch_size: int = 32

class TestWandbIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_wandb_run = Mock()
        self.mock_wandb_run.project = "test_experiment"
        self.mock_wandb_run.id = "test_id"

    def test_wandb_logging(self):
        """Test logging with wandb integration"""
        logger = Logger(
            experiment_name="test_experiment",
            wandb_run=self.mock_wandb_run
        )
        metrics = {"loss": 0.5, "accuracy": 0.95}
        logger.log(metrics)
        
        self.mock_wandb_run.log.assert_called_once_with(metrics, step=0)

    @patch('wandb.init')
    def test_logger_from_args_with_wandb(self, mock_wandb_init):
        """Test creating logger from args with wandb enabled"""
        mock_wandb_init.return_value = self.mock_wandb_run
        
        args = LoggerArgs(experiment_name="test_experiment")
        hparams = MockHyperParams()  # Using the dataclass instead of Mock
        
        logger = Logger.from_args(args, hparams, use_wandb=True)
        self.assertEqual(logger.wandb_run, self.mock_wandb_run)
        mock_wandb_init.assert_called_once_with(
            project="test_experiment",
            config={"learning_rate": 0.01, "batch_size": 32}
        )

    def test_logger_from_args_multiple_hparams(self):
        """Test creating logger with multiple hyperparameter objects"""
        @dataclass
        class TrainingParams:
            epochs: int = 100
            optimizer: str = "adam"

        @dataclass
        class ModelParams:
            hidden_size: int = 256
            num_layers: int = 2

        args = LoggerArgs(experiment_name="test_experiment")
        hparams_list = [
            MockHyperParams(),
            TrainingParams(),
            ModelParams()
        ]

        logger = Logger.from_args(args, hparams_list, use_wandb=False)
        
        # Check if all parameters are in the hyper_params_dict
        expected_params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "hidden_size": 256,
            "num_layers": 2
        }
        self.assertEqual(logger.hyper_params_dict, expected_params)


if __name__ == '__main__':
    unittest.main(verbosity=2)