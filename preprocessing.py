import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Memuat file CSV dataset ke dalam DataFrame Pandas.
    """
    print(f"Memuat data dari: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Membersihkan dataset dengan menghapus baris yang mengandung nilai kosong (missing values).
    """
    print("Membersihkan data dari missing values...")
    df_cleaned = df.dropna().copy()
    print(f"Jumlah baris sebelum dibersihkan: {len(df)}")
    print(f"Jumlah baris setelah dibersihkan: {len(df_cleaned)}")
    return df_cleaned

def transform_data(df, label_cols=['Team', 'Ban', 'Role Ban', 'Pick', 'Role Pick']):
    """
    Melakukan Label Encoding pada kolom kategorikal yang ditentukan.
    """
    print(f"Melakukan Label Encoding pada kolom: {label_cols}")
    df_encoded = df.copy()
    encoders = {}
    for col in label_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    return df_encoded, encoders

def split_dataset(df, target_column='MVP', test_size=0.2, random_state=42):
    """
    Membagi dataset menjadi set data latih (train) dan data uji (test).
    Target default diset ke 'MVP' sebagai contoh target klasifikasi.
    """
    print(f"Membagi dataset dengan target: '{target_column}' dan porsi test: {test_size}")

    # Memisahkan fitur dan target
    # Mengecualikan beberapa fitur non-numerik / objek agar siap dimasukkan ke model
    numeric_df = df.select_dtypes(include=['number'])

    if target_column not in numeric_df.columns:
        raise ValueError(f"Target column '{target_column}' tidak ditemukan dalam kolom numerik.")

    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]

    # Gunakan stratify=y untuk memastikan proporsi target seimbang antara data train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Ukuran X_train: {X_train.shape}, Ukuran X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
