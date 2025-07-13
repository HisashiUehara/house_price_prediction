import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# データの読み込み
df = pd.read_csv('data/Tokyo_Bunkyo Ward_Koishikawa_20243_20244.csv', encoding='cp932')

print("=== ニューラルネットワークモデルの構築 ===")
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

# データの標準化（ニューラルネットワークでは重要）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# ニューラルネットワークモデルの作成
model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),  # 3層の隠れ層
    activation='relu',                  # 活性化関数
    solver='adam',                      # 最適化アルゴリズム
    alpha=0.001,                        # 正則化パラメータ
    learning_rate='adaptive',           # 学習率の調整
    max_iter=1000,                      # 最大反復回数
    random_state=42
)

print("\n=== モデルの学習 ===")
model.fit(X_train_scaled, y_train_scaled)

# テストデータでの予測
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

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
    # データを標準化
    area_scaled = scaler_X.transform([[area]])
    # 予測
    pred_scaled = model.predict(area_scaled)
    # 逆変換
    predicted_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    print(f"面積{area}㎡ → 予測価格: {predicted_price:,.0f}円")

print("\n=== 線形回帰との比較 ===")
print("ニューラルネットワークの利点:")
print("1. 非線形な関係性を学習できる")
print("2. 複雑なパターンを捉えられる")
print("3. データの特徴を自動的に学習")

print("\n=== 制限事項 ===")
print("1. データが少ない（15件）ため、学習が不十分")
print("2. 過学習のリスクがある")
print("3. より多くのデータと特徴量が必要")

print("\n=== 次のステップ ===")
print("1. より多くのデータを収集")
print("2. 追加の特徴量を検討（築年数、駅距離など）")
print("3. ハイパーパラメータの調整")
print("4. 異なるアーキテクチャの試行") 