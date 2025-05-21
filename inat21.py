import os
import pandas as pd
from .utils import Datum, DatasetBase

template = ['a photo of a {}.']

class INat21(DatasetBase):
    dataset_dir = 'inat21'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # CSV paths for train and test splits
        self.train_csv = os.path.join(self.dataset_dir, f"train_k{num_shots}.csv")
        self.test_csv = os.path.join(self.dataset_dir, "test.csv")

        self.template = template

        # Load CSV dataframes
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        # Build unified label2id mapping for train+test classes
        unique_labels = sorted(set(train_df['label']).union(set(test_df['label'])))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        # Function to fix image paths (avoid duplicated folders)
        def fix_path(image_path):
            # Normalize separators to '/'
            normalized = image_path.replace("\\", "/")
            
            # If already starts with double train_mini, don't prepend anything
            if normalized.startswith('train_mini/train_mini/'):
                return os.path.join(self.dataset_dir, normalized)
            
            # If starts with one train_mini/, join directly
            if normalized.startswith('train_mini/'):
                return os.path.join(self.dataset_dir, normalized)
            
            # Otherwise prepend train_mini/
            return os.path.join(self.dataset_dir, 'train_mini', normalized)

        # Create Datum instances for train and test sets
        train_data = [
            Datum(
                impath=fix_path(row['image_path']),
                label=label2id[row['label']],
                classname=row['label']
            )
            for _, row in train_df.iterrows()
        ]

        test_data = [
            Datum(
                impath=fix_path(row['image_path']),
                label=label2id[row['label']],
                classname=row['label']
            )
            for _, row in test_df.iterrows()
        ]

        # Generate few-shot training subset with num_shots
        train_x = self.generate_fewshot_dataset(train_data, num_shots=num_shots)

        # Use test_data as validation (or you can split test further if desired)
        val = test_data

        self._classnames = unique_labels


        # Initialize parent DatasetBase
        super().__init__(train_x=train_x, val=val, test=test_data)
