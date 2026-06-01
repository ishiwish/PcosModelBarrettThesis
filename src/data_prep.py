import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
 
os.makedirs("figures", exist_ok=True)
 
df = pd.read_csv("data/pcos_cleaned.csv")
 
# feature groups
categorical = [
    "Menstrual_Irregularity", "Alopecia", "Skin_Darkening_Acanthosis",
    "Smoking_Status", "Alcohol_Intake", "Physical_Activity_Level",
    "Dietary_Sugar_Intake", "Acne_Severity", "PCOS_Diagnosis",
]
lifestyle = [
    "Age", "BMI", "Waist_Hip_Ratio", "Physical_Activity_Level",
    "Smoking_Status", "Alcohol_Intake", "Dietary_Sugar_Intake", "Sleep_Hours",
    "Menstrual_Irregularity", "Menstrual_Cycle_Length_days", "Hirsutism_Score_FG",
    "Acne_Severity", "Alopecia", "Skin_Darkening_Acanthosis",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Gravidity", "Parity", "Age_at_Menarche",
]
 
#Fig 1: full correlation heatmap
corr_full = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr_full, dtype=bool)) 
 
fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(corr_full, mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, linewidths=0.3, linecolor="white",
            annot=False, ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title("Pearson Correlation Matrix — All Features", fontsize=14, pad=12)
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)
plt.tight_layout()
plt.savefig("figures/fig1_full_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")
 
# point-biserial correlations, lifestyle vs PCOS Diagnosis 
results = []
for col in lifestyle:
    r, p = stats.pointbiserialr(df["PCOS_Diagnosis"], df[col])
    results.append({"feature": col, "r": r, "p_value": p, "sig": p < 0.05})
 
pb_df = pd.DataFrame(results).sort_values("r", ascending=True)
colors = ["#E24B4A" if s else "#B4B2A9" for s in pb_df["sig"]]
 
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(pb_df["feature"], pb_df["r"], color=colors, height=0.6)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Point-biserial r", fontsize=11)
ax.set_title("Lifestyle Feature Correlations with PCOS Diagnosis", fontsize=13, pad=10)
ax.tick_params(labelsize=9)
 
for i, (_, row) in enumerate(pb_df.iterrows()):
    if row["sig"]:
        x = row["r"] + 0.002 if row["r"] >= 0 else row["r"] - 0.002
        ax.text(x, i, "* p<0.05", va="center", fontsize=7.5,
                ha="left" if row["r"] >= 0 else "right", color="#E24B4A")
 
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#E24B4A", label="p < 0.05"),
                   Patch(color="#B4B2A9", label="p ≥ 0.05")],
          fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig2_pointbiserial_lifestyle_vs_pcos.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")
 
# Lifestyle inter-feature heatmap
corr_lifestyle = df[lifestyle].corr()
mask = np.triu(np.ones_like(corr_lifestyle, dtype=bool))
 
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_lifestyle, mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5,
            linecolor="white", ax=ax, annot_kws={"size": 8},
            cbar_kws={"shrink": 0.7})
ax.set_title("Lifestyle Feature Inter-Correlations", fontsize=13, pad=12)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("figures/fig3_lifestyle_intercorrelation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")
 
# multicollinearity check flag any pairs with |r| > 0.7 — drop one of them before modeling
all_num = df.select_dtypes(include=np.number).columns.tolist()
corr_all = df[all_num].corr()
 
high_corr = [
    (all_num[i], all_num[j], round(corr_all.iloc[i, j], 3))
    for i in range(len(all_num))
    for j in range(i + 1, len(all_num))
    if abs(corr_all.iloc[i, j]) > 0.7
]
 
print("\nMulticollinearity check (|r| > 0.7):")
if high_corr:
    for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {a} <-> {b}: r = {r}")
else:
    print("  None — no multicollinearity issues.")
 
#summary
print("\nCorrelation summary (lifestyle features vs PCOS_Diagnosis):")
print(f"{'Feature':<35} {'r':>7} {'p':>10} {'':>4}")
print("-" * 58)
for _, row in pb_df.sort_values("r", ascending=False).iterrows():
    sig = "*" if row["sig"] else ""
    print(f"{row['feature']:<35} {row['r']:>7.4f} {row['p_value']:>10.4f} {sig:>4}")
 