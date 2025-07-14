from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.feature_store import RedistFeatureStore
from config.paths_config import TRAIN_PATH, TEST_PATH,RAW_DIR
from config.database_config import DB_CONFIG



data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
data_ingestion.run()

feature_store = RedistFeatureStore()
data_processor = DataProcessing(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    feature_store=feature_store
)
data_processor.run()

model_trainer = ModelTraining(feature_store=feature_store)
model_trainer.run()
print("Model training completed successfully.")