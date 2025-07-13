import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedistFeatureStore

from src.logger import get_logger
from src.custom_exception import CustomException

from config.paths_config import TRAIN_PATH, TEST_PATH

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,train_path, test_path,feature_store : RedistFeatureStore):
        self.train_path = train_path
        self.test_path = test_path

        self.data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

        self.feature_store = feature_store
        logger.info("DataProcessing initialized with train and test paths.")
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}",e)

    def preprocess_data(self):
        try:
            
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
            self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
            self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes

            self.data['Familysize'] = self.data['SibSp'] + self.data['Parch'] + 1

            self.data['Isalone'] = (self.data['Familysize'] == 1).astype(int)
            self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)
            self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map(
                {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
            ).fillna(4)
            self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
            self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']

            logger.info("Data preprocessing completed successfully.")
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise CustomException(f"Error in data preprocessing: {e}", e)

    def handle_imbalanced_data(self):
        try:
            X = self.data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
            y = self.data['Survived']
            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)
            logger.info("Imbalanced data handled using SMOTE.")

        except Exception as e:
            logger.error(f"Error handling imbalanced data: {e}")
            raise CustomException(f"Error handling imbalanced data: {e}", e)
    
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for index, row in self.data.iterrows():
                entity_id = row['PassengerId']
                features = {
                    "Age" : row["Age"],
                    "Fare" : row["Fare"],
                    "Sex" : row["Sex"],
                    "Embarked" : row["Embarked"],
                    "Familysize" : row["Familysize"],
                    "Isalone" : row["Isalone"],
                    "HasCabin" : row["HasCabin"],
                    "Title" : row["Title"],
                    "Pclass_Fare" : row["Pclass_Fare"],
                    "Age_Fare" : row["Age_Fare"],
                    "Survived" : row["Survived"],
                    "Pclass" : row["Pclass"]
                }
                batch_data[entity_id] = features
            self.feature_store.store_features_batch(batch_data)
            logger.info("Features stored in Redis successfully.")
        except Exception as e:
            logger.error(f"Error storing features in Redis: {e}")
            raise CustomException(f"Error storing features in Redis: {e}", e)
    
    def retrive_feature_redis_store(self,entity_id):
        try:
            features = self.feature_store.get_feature(entity_id)
            logger.info("Features retrieved from Redis successfully.")
            return features
        except Exception as e:
            logger.error(f"Error retrieving features from Redis: {e}")
            raise CustomException(f"Error retrieving features from Redis: {e}", e)
    
    def run(self):
        logger.info("Starting data processing pipeline...")
        self.load_data()
        self.preprocess_data()
        self.handle_imbalanced_data()
        self.store_feature_in_redis()
        logger.info("Data processing pipeline completed successfully.")
    
if __name__ == "__main__":
    feature_store = RedistFeatureStore()
    data_processor = DataProcessing(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        feature_store=feature_store
    )
    data_processor.run()

    print(data_processor.retrive_feature_redis_store(734))    
