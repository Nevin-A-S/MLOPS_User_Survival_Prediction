import redis
import json

class RedistFeatureStore:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.StrictRedis(host=host, port=port, db=db,decode_responses=True)

    def store_feature(self, entity_id,features):
        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_feature(self, entity_id):
        key = f"entity:{entity_id}:features"
        feature_data = self.client.get(key)
        if feature_data:
            return json.loads(feature_data)
        else:
            return None
    
    def store_features_batch(self, batch_data):
        for entity_id, features in batch_data.items():
            self.store_feature(entity_id, features)
    
    def get_features_batch(self, entity_ids):
        batch_features = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_feature(entity_id)
        return batch_features
    
    def get_all_entity_ids(self):
        keys = self.client.keys("entity:*:features")
        entity_ids = [key.split(":")[1] for key in keys]
        return entity_ids