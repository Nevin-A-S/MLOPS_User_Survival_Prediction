from src.logger import get_logger
from src.custom_exception import CustomException

import pandas as pd
from src.feature_store import RedistFeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

import os
import pickle

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, feature_store: RedistFeatureStore,model_save_path='artifacts/models'):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None
        os.makedirs(self.model_save_path, exist_ok=True)

        logger.info("ModelTraining initialized with feature store and model save path.")
        
    
    def load_data_from_redis(self,entity_ids):
        try:
            logger.info("Extracting features from Redis")
            data = []

            for entity_id in entity_ids:
                features = self.feature_store.get_feature(entity_id)
                if features :
                    features['entity_id'] = entity_id
                    data.append(features)
                else:
                    logger.warning(f"No features found for entity_id: {entity_id}")

            logger.info("Data loaded from Redis successfully.")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data from Redis: {e}")
            raise CustomException(f"Error loading data from Redis: {e}", e)
    
    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()

            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)
            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_Train = train_df.drop(columns=['Survived'] ,axis=1)
            logger.info(f"X_Train shape: {X_Train.shape}")
            logger.info(f"X_Train columns: {X_Train.columns.tolist()}")

            y_Train = train_df['Survived']
            logger.info(f"y_Train shape: {y_Train.shape}")
            
            X_Test = test_df.drop(columns=['Survived'], axis=1)
            logger.info(f"X_Test shape: {X_Test.shape}")
            logger.info(f"X_Test columns: {X_Test.columns.tolist()}")
            
            y_Test = test_df['Survived']
            logger.info(f"y_Test shape: {y_Test.shape}")

            return X_Train, y_Train, X_Test, y_Test
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException(f"Error preparing data: {e}", e)
    
    def hyperparameter_tuning(self, X_train, y_train):
        
        try:
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train, y_train)

            best_rf = random_search.best_estimator_
            
            logger.info(f"Best Random Forest parameters: {random_search.best_params_}")

            return best_rf
        
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise CustomException(f"Error in hyperparameter tuning: {e}", e)
    
    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model training and evaluation...")
            self.model = self.hyperparameter_tuning(X_train, y_train)

            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained successfully with accuracy: {accuracy}")

            self.save_model()
            return accuracy
        
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {e}")
            raise CustomException(f"Error in model training and evaluation: {e}", e)
    
    def save_model(self):
        try:
            if self.model is not None:
                model_path = os.path.join(self.model_save_path, 'random_forest_model.pkl')
                with open(model_path, 'wb') as model_file:
                    pickle.dump(self.model, model_file)
                logger.info(f"Model saved successfully at {model_path}")
            else:
                logger.warning("No model to save.")
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException(f"Error saving model: {e}", e)
    
    def run(self):
        try:
            logger.info("Starting model training pipeline...")
            X_train, y_train, X_test, y_test = self.prepare_data()
            accuracy = self.train_and_evaluate_model(X_train, y_train, X_test, y_test)
            logger.info(f"Model training pipeline completed with accuracy: {accuracy}")
        
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException(f"Error in model training pipeline: {e}", e)
if __name__ == "__main__":
    feature_store = RedistFeatureStore()
    model_trainer = ModelTraining(feature_store=feature_store)
    model_trainer.run()
    print("Model training completed successfully.")