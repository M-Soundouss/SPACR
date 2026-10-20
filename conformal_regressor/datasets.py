import numpy as np
import pandas as pd
import torch
import cv2
import random
from pathlib import Path
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from datasets import load_dataset

# ---------------------------------------------------------
# 1. TABULAR DATASETS
# ---------------------------------------------------------

def _prepare_general_tabular(data_id, target_col, batch_size, seed, num_workers, log_target=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    print(f"Loading OpenML Dataset ID: {data_id}...")

    # 1. Load Data
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        df = data.frame
    except Exception as e:
        raise RuntimeError(f"Failed to fetch OpenML ID {data_id}: {e}")

    if target_col is None:
        target_col = data.target_names[0]

    df = df.dropna(subset=[target_col])

    # 2. Separate X and y
    if log_target:
        y = np.log(df[target_col].to_numpy().astype(np.float32) + 1)
    else:
        y = df[target_col].to_numpy().astype(np.float32)

    X = df.drop(columns=[target_col])

    # 3. Split Data: 60% Train, 20% Val, 20% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=seed
    )

    # 4. Preprocessing Pipeline (Strict Separation)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Numeric: Impute missing with mean -> StandardScale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute missing with 'missing' -> OneHot
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    input_dim = X_train_processed.shape[1]
    print(f"Final Input Dimension: {input_dim}")

    # 5. Create Tensors
    train_dataset = TensorDataset(torch.tensor(X_train_processed), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val_processed), torch.tensor(y_val).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test_processed), torch.tensor(y_test).unsqueeze(1))

    # 6. Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return input_dim, train_loader, val_loader, test_loader


def prepare_bikesharing_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(42712, "count", batch_size, seed, num_workers, log_target=True)


def prepare_brazilian_houses_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(42688, "total_(BRL)", batch_size, seed, num_workers, log_target=True)


def prepare_california_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(44025, "price", batch_size, seed, num_workers, log_target=True)


def prepare_cpu_act_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(197, "usr", batch_size, seed, num_workers, log_target=True)


def prepare_diamonds_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(42225, "price", batch_size, seed, num_workers, log_target=True)


def prepare_fifa_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(44026, "wage_eur", batch_size, seed, num_workers, log_target=True)


def prepare_house_sales_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(42731, "price", batch_size, seed, num_workers, log_target=True)


def prepare_isolet_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(44135, "class", batch_size, seed, num_workers, log_target=True)


def prepare_medical_charges_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(42720, "Average_Total_Payments", batch_size, seed, num_workers, log_target=True)


def prepare_pol_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(201, "foo", batch_size, seed, num_workers, log_target=True)


def prepare_superconduct_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(44148, "criticaltemp", batch_size, seed, num_workers, log_target=True)


def prepare_winequality_tabular_data(batch_size=32, seed=42, num_workers=0):
    return _prepare_general_tabular(287, "quality", batch_size, seed, num_workers, log_target=True)


# ---------------------------------------------------------
# 2. IMAGE DATASETS (Drift, UTKFace)
# ---------------------------------------------------------

class TreeImageDataset(Dataset):
    def __init__(self, dataframe, img_dir: Path, transform=None):
        self.labels_df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = str(self.img_dir / self.labels_df.iloc[idx]["filename"])

        img_array = cv2.imread(img_path)
        if img_array is None:
            raise ValueError(f"OpenCV failed to load image at: {img_path}")

        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_array)

        tree_count = self.labels_df.iloc[idx]["tree_count"]
        tree_count = torch.tensor([tree_count], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, tree_count


def _load_drift_country_data(country):
    base_dir = Path(__file__).parents[1] / "datasets" / "f1Hmpeh6O2" / "DRIFT" / country
    label_file = base_dir / "labels.csv"
    img_dir = base_dir / "images"

    try:
        df = pd.read_csv(label_file)
    except Exception as e:
        print(f"Error reading label file {label_file}: {e}")
        raise

    return df, img_dir


def prepare_trees_data_generic(country, batch_size=32, img_size=224, num_workers=0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed()
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    df, img_dir = _load_drift_country_data(country)

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=seed)

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "eval": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_ds = TreeImageDataset(train_df.reset_index(drop=True), img_dir, data_transforms["train"])
    val_ds = TreeImageDataset(val_df.reset_index(drop=True), img_dir, data_transforms["eval"])
    test_ds = TreeImageDataset(test_df.reset_index(drop=True), img_dir, data_transforms["eval"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                              generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def prepare_trees_data(batch_size=32, img_size=224, num_workers=0, seed=42):
    return prepare_trees_data_generic("Denmark", batch_size, img_size, num_workers, seed)


class UTKFaceHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, img_col="image", key_col="__key__"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.img_col = img_col
        self.key_col = key_col

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.hf_dataset[idx]
        try:
            image = row[self.img_col].convert("RGB")
            if self.transform: image = self.transform(image)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.key_col in row and row[self.key_col] is not None:
            filename = str(row[self.key_col]).split('/')[-1]
        else:
            return self.__getitem__(random.randint(0, len(self) - 1))

        try:
            age = int(filename.split('_')[0])
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return image, torch.tensor([age], dtype=torch.float32)


def prepare_utkface_data(batch_size=32, img_size=224, num_workers=0, seed=42):
    print("Loading UTKFace...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(id):
        torch.manual_seed(torch.initial_seed())

    ds = load_dataset("py97/UTKFace-Cropped", split="train")

    if len(ds) == 0:
        raise RuntimeError("UTKFace dataset is empty on load!")

    sample = ds[0]
    print(f"DEBUG: UTKFace columns found: {ds.column_names}")

    img_col = "image" if "image" in sample else None
    if not img_col:
        for k, v in sample.items():
            if hasattr(v, "convert"):
                img_col = k
                break

    key_col = None
    candidates = ["__key__", "file_name", "filename", "id"]
    for c in candidates:
        if c in sample:
            key_col = c
            break

    if not key_col:
        for k, v in sample.items():
            if isinstance(v, str) and "_" in v:
                key_col = k
                break

    if not img_col or not key_col:
        raise RuntimeError(
            f"Could not auto-detect image/filename columns in UTKFace. Available columns: {ds.column_names}")

    print(f"DEBUG: Auto-detected Image Column: '{img_col}', Key Column: '{key_col}'")

    def is_valid(x):
        if x[img_col] is None: return False
        try:
            val = x[key_col]
            if val is None: return False
            fname = str(val).split('/')[-1]
            int(fname.split('_')[0])
            return True
        except:
            return False

    before_len = len(ds)
    ds = ds.filter(is_valid)
    after_len = len(ds)
    print(f"UTKFace Filtering: {before_len} -> {after_len} images.")

    if after_len == 0:
        raise RuntimeError(
            "UTKFace dataset is empty after filtering! The 'is_valid' logic likely failed to match the dataset structure.")

    split1 = ds.train_test_split(test_size=0.4, seed=seed)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)

    tf_train = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_eval = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tr = DataLoader(UTKFaceHuggingFaceDataset(split1["train"], tf_train, img_col, key_col), batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, generator=generator)
    va = DataLoader(UTKFaceHuggingFaceDataset(split2["train"], tf_eval, img_col, key_col), batch_size=batch_size,
                    num_workers=num_workers)
    te = DataLoader(UTKFaceHuggingFaceDataset(split2["test"], tf_eval, img_col, key_col), batch_size=batch_size,
                    num_workers=num_workers)

    return tr, va, te


if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_trees_data(batch_size=32)