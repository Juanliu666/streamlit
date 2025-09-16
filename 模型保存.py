import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Permute, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import joblib  # 用于保存标准化器
import os

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 读取Excel数据
file_path = r"C:\Users\lenovo\Desktop\巴塞尔公约亚太区域中心\30.物质-能量协同处置单元交互作用智能算法研究-机器学习\3 专利\专利数据-20250819-加多.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# 定义输入输出列
input_features = ['污泥添加比例（%）', 'C含量（%）', '热解温度（℃）']
output_features = [
    '液体产率（%）', '气体产率（%）', '气体中CO2含量（%）', '气体中CH4含量（%）',
    '热解油中酸含量（%）', '热解油中酚含量（%）', '热解油中含氮化合物含量（%）'
]

# 提取输入和输出数据
X = data[input_features].values
y = data[output_features].values

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 重塑数据为LSTM输入格式 (样本数, 时间步长=1, 特征数)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# 自定义注意力机制层
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)  # (batch_size, features, time_steps)
    a = Dense(time_steps, activation='softmax')(a)  # 为每个特征学习权重
    a = Permute((2, 1), name='attention_weights')(a)  # 恢复原始维度
    output = Multiply()([inputs, a])  # 应用注意力权重
    return output

# 构建Attention-LSTM模型
def build_attention_lstm(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    # LSTM层
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.4)(lstm_out)
    # 注意力机制
    attention_out = attention_block(lstm_out, input_shape[0])
    # 展平后连接全连接层
    flat = Flatten()(attention_out)
    dense = Dense(128, activation='relu')(flat)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation='relu')(dense)
    # 多输出层
    outputs = Dense(output_dim)(dense)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    return model

# 划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_scaled, test_size=0.2, random_state=42
)

# 构建并训练Attention-LSTM模型
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_attention_lstm(input_shape, y_train.shape[1])
print("开始训练Attention-LSTM模型...")
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=8,
    verbose=1
)

# 保存模型
model.save('attention_lstm_model.h5')
print("模型已保存为 'attention_lstm_model.h5'")

# 保存标准化器
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("标准化器已保存")

# 模型评估
y_pred_scaled_train= model.predict(X_train)
y_pred_train = scaler_y.inverse_transform(y_pred_scaled_train)
y_train_original = scaler_y.inverse_transform(y_train)
y_pred_scaled_test= model.predict(X_test)
y_pred_test = scaler_y.inverse_transform(y_pred_scaled_test)
y_test_original = scaler_y.inverse_transform(y_test)

# 计算整体评估指标
total_r2_train= r2_score(y_train_original, y_pred_train)
total_mse_train = mean_squared_error(y_train_original, y_pred_train)
total_r2_test= r2_score(y_test_original, y_pred_test)
total_mse_test = mean_squared_error(y_test_original, y_pred_test)

# 计算NMSE (归一化均方误差)
variance_train = np.var(y_train_original, axis=0)
mse_per_target_train = mean_squared_error(y_train_original, y_pred_train, multioutput='raw_values')
nmse_per_target_train = mse_per_target_train / variance_train
total_nmse_train = np.mean(nmse_per_target_train)

variance_test = np.var(y_test_original, axis=0)
mse_per_target_test = mean_squared_error(y_test_original, y_pred_test, multioutput='raw_values')
nmse_per_target_test = mse_per_target_test / variance_test
total_nmse_test = np.mean(nmse_per_target_test)

print(f"训练集整体评估结果:")
print(f"R²系数: {total_r2_train:.4f}")
print(f"均方误差(MSE): {total_mse_train:.4f}")
print(f"归一化均方误差(NMSE): {total_nmse_train:.4f}")
print("=" * 50)
print(f"测试集整体评估结果:")
print(f"R²系数: {total_r2_test:.4f}")
print(f"均方误差(MSE): {total_mse_test:.4f}")
print(f"归一化均方误差(NMSE): {total_nmse_test:.4f}")
print("=" * 50)