from pipelines.training_pipeline import train_pipeline
from zenml.client import Client



if __name__ == "main":
    """run the pipeline"""
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")