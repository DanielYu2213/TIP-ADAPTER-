import pandas as pd

def fix_csv_paths(csv_path):
    df = pd.read_csv(csv_path)

    def fix_path(p):
        # Normalize slashes
        p = p.replace("\\", "/")
        # Remove all leading 'train_mini/' prefixes
        while p.startswith('train_mini/'):
            p = p[len('train_mini/'):]
        # Prepend exactly one 'train_mini/'
        return f"train_mini/{p}"

    df['image_path'] = df['image_path'].apply(fix_path)
    df.to_csv(csv_path, index=False)
    print(f"Fixed paths in: {csv_path}")

# Replace with your actual paths
fix_csv_paths("data/inat21/train_k8.csv")
fix_csv_paths("data/inat21/test.csv")
