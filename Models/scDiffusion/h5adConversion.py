import pandas as pd

# 1) Load CSV
path = "data/suzuki_original-expression.csv"   # change this
df = pd.read_csv(path)

print("\n✅ Loaded CSV")
print("Shape (rows, cols):", df.shape)

# 2) Show first few rows
print("\n🔹 First 5 rows:")
print(df.head())

# 3) Show column names
print("\n🔹 Column names:")
print(df.columns.tolist()[:30], "...")  # first 30 only

# 4) Check if there's an ID-like first column
first_col = df.columns[0]
print("\n🔹 First column name:", first_col)
print("Example values:", df[first_col].head().tolist())

# 5) Check missing values
missing = df.isna().sum().sum()
print("\n🔹 Total missing values in entire CSV:", missing)

# 6) Check data types summary
print("\n🔹 Data types count:")
print(df.dtypes.value_counts())

# 7) Try to guess if there is a label column
possible_label_cols = []
for col in df.columns:
    # labels often are non-numeric or small unique count
    unique_count = df[col].nunique(dropna=True)
    if unique_count < 50 and df[col].dtype == "object":
        possible_label_cols.append(col)

print("\n🔹 Possible label columns (guess):", possible_label_cols)

# 8) Check numeric-only matrix size
numeric_df = df.select_dtypes(include="number")
print("\n🔹 Numeric-only matrix shape:", numeric_df.shape)

# 9) Simple heuristic: if numeric cols are huge and rows are small, maybe transpose needed
rows, cols = numeric_df.shape
if cols > rows * 10:
    print("\n⚠️ Heuristic: You have many more columns than rows.")
    print("This is normal for genes/features, but if you expected cells as columns, you might need transpose (df.T).")
else:
    print("\n✅ Orientation seems fine (rows=samples, cols=features) for most ML tasks.")
