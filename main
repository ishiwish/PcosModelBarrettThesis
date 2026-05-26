import pandas as pd
import numpy as np
import os
 
os.makedirs("data", exist_ok=True)
 
df = pd.read_excel("/Users/amritaupadhyay/IshitaProjects/PCOS PROJECT/PCOS Dataset (1).xlsx")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
 
# Fix BMI values below 12 
bad_idx = df[df["BMI"] < 12].index
for i in bad_idx:
    fixed = df.loc[i, "Weight_kg"] / (df.loc[i, "Height_cm"] / 100) ** 2
    print(f"  BMI row {i}: {df.loc[i, 'BMI']:.2f} → {fixed:.2f}")
    df.loc[i, "BMI"] = round(fixed, 2)
 
#Replace negative values with median
non_negative = [
    "LH_mIU_mL", "LH_FSH_Ratio", "Free_Testosterone_pg_mL", "DHEAS_ug_dL",
    "Prolactin_ng_mL", "Estradiol_pg_mL", "Progesterone_ng_mL", "SHBG_nmol_L",
    "Fasting_Insulin_uIU_mL", "HOMA_IR", "Triglycerides_mg_dL",
    "Ovary_Volume_Right_cm3", "CRP_mg_L", "ALT_U_L", "AST_U_L",
    "TSH_uIU_mL", "Vitamin_D_ng_mL",
]
 
total_replaced = 0
for col in non_negative:
    mask = df[col] < 0
    n = mask.sum()
    if n > 0:
        df.loc[mask, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
        print(f"  {col}: {n} negatives → median")
        total_replaced += n
 
print(f"  Total replaced: {total_replaced} values across {len(non_negative)} columns")
 
#Winsorize outliers at +/-3 standard deviations
categorical = [
    "Menstrual_Irregularity", "Alopecia", "Skin_Darkening_Acanthosis",
    "Smoking_Status", "Alcohol_Intake", "Physical_Activity_Level",
    "Dietary_Sugar_Intake", "Acne_Severity", "PCOS_Diagnosis",
]
continuous = [c for c in df.select_dtypes(include=np.number).columns if c not in categorical]
 
for col in continuous:
    mean, std = df[col].mean(), df[col].std()
    lower, upper = mean - 3 * std, mean + 3 * std
    n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_capped > 0:
        df[col] = df[col].clip(lower, upper)
        print(f"  {col}: {n_capped} value(s) capped")
 
# Validation
print("\nValidation:")
print(f"  Missing values : {df.isnull().sum().sum()}  (expected 0)")
print(f"  Negatives left : {sum((df[c] < 0).sum() for c in non_negative)}  (expected 0)")
print(f"  Duplicate rows : {df.duplicated().sum()}  (expected 0)")
print(f"  BMI range      : {df['BMI'].min():.1f} – {df['BMI'].max():.1f}")
print(f"  Final shape    : {df.shape}")
 
df.to_csv("data/pcos_cleaned.csv", index=False)
print("\nSaved → data/pcos_cleaned.csv")


