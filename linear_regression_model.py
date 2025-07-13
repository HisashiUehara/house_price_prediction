import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
df = pd.read_csv('data/Tokyo_Bunkyo Ward_Koishikawa_20243_20244.csv', encoding='cp932')

print("=== 線形回帰モデルの構築 ===")
print(f"データサイズ: {df.shape}")

# 特徴量（X）と目的変数（y）の準備
X = df[['面積（㎡）']]  # 特徴量：面積
y = df['取引価格（総額）']  # 目的変数：価格

print(f"特徴量（面積）の形状: {X.shape}")
print(f"目的変数（価格）の形状: {y.shape}")

# データを訓練用とテスト用に分割（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n訓練データ: {X_train.shape[0]}件")
print(f"テストデータ: {X_test.shape[0]}件")

# 線形回帰モデルの作成と訓練
model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== モデルの学習結果 ===")
print(f"傾き（係数）: {model.coef_[0]:,.0f}")
print(f"切片: {model.intercept_:,.0f}")

# モデルの数式
print(f"\n予測式: 価格 = {model.coef_[0]:,.0f} × 面積 + {model.intercept_:,.0f}")

# テストデータでの予測
y_pred = model.predict(X_test)

# モデルの評価
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== モデルの評価 ===")
print(f"平均二乗誤差（MSE）: {mse:,.0f}")
print(f"平均二乗誤差の平方根（RMSE）: {rmse:,.0f} 円")
print(f"決定係数（R²）: {r2:.3f}")

# 決定係数の解釈
if r2 > 0.7:
    interpretation = "良い予測精度"
elif r2 > 0.5:
    interpretation = "中程度の予測精度"
elif r2 > 0.3:
    interpretation = "低い予測精度"
else:
    interpretation = "予測精度が低い"

print(f"予測精度の評価: {interpretation}")

# 実際の値と予測値の比較
print("\n=== 実際の値 vs 予測値 ===")
for i in range(len(y_test)):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    area = X_test.iloc[i]['面積（㎡）']
    error = abs(actual - predicted)
    error_rate = (error / actual) * 100
    
    print(f"面積{area}㎡: 実際{actual:,.0f}円 → 予測{predicted:,.0f}円 (誤差{error_rate:.1f}%)")

print("\n=== 新しいデータでの予測テスト ===")
# 新しい面積での予測
new_areas = [50, 80, 120, 200]
for area in new_areas:
    predicted_price = model.predict([[area]])[0]
    print(f"面積{area}㎡ → 予測価格: {predicted_price:,.0f}円")

print("\n=== 次のステップ ===")
print("モデルの改善案:")
print("1. より多くの特徴量を追加（築年数、駅距離など）")
print("2. より多くのデータを収集")
print("3. 異なる機械学習手法を試す（ランダムフォレストなど）") 