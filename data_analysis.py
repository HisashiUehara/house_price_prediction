import pandas as pd
import numpy as np

# データの読み込み
df = pd.read_csv('data/Tokyo_Bunkyo Ward_Koishikawa_20243_20244.csv', encoding='cp932')

print("=== データ分析結果 ===")
print(f"データの形状: {df.shape}")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")

print("\n=== 面積と価格の基本統計 ===")
print(df[['面積（㎡）', '取引価格（総額）']].describe())

print("\n=== 相関分析 ===")
# 相関係数の計算
correlation = df['面積（㎡）'].corr(df['取引価格（総額）'])
print(f"面積と価格の相関係数: {correlation:.3f}")

# 相関係数の解釈
if correlation > 0.7:
    strength = "強い正の相関"
elif correlation > 0.3:
    strength = "中程度の正の相関"
elif correlation > 0:
    strength = "弱い正の相関"
elif correlation > -0.3:
    strength = "ほとんど相関なし"
elif correlation > -0.7:
    strength = "弱い負の相関"
else:
    strength = "強い負の相関"

print(f"相関の強さ: {strength}")

print("\n=== データの詳細 ===")
print("面積の範囲:")
print(f"  最小: {df['面積（㎡）'].min()} ㎡")
print(f"  最大: {df['面積（㎡）'].max()} ㎡")
print(f"  平均: {df['面積（㎡）'].mean():.1f} ㎡")

print("\n価格の範囲:")
print(f"  最小: {df['取引価格（総額）'].min():,} 円")
print(f"  最大: {df['取引価格（総額）'].max():,} 円")
print(f"  平均: {df['取引価格（総額）'].mean():,.0f} 円")

print("\n=== 次のステップ ===")
print("このデータを使って線形回帰モデルを作成できます。")
print("面積を特徴量、価格を目的変数として予測モデルを構築しましょう。") 