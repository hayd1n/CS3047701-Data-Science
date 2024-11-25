import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from collections import defaultdict


class FraudDetectionModel:
    def __init__(self):
        self.label_encoders = {}
        self.label_maps = defaultdict(dict)  # 儲存每個特徵的類別映射
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=200,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric=["error", "logloss"],
            enable_categorical=True,
        )

    def safe_label_encode(self, series, feature_name, is_training=True):
        """安全的標籤編碼，處理未見過的類別"""
        if is_training:
            # 訓練階段：創建新的編碼器和映射
            unique_values = series.unique()
            self.label_maps[feature_name] = {
                val: i for i, val in enumerate(unique_values)
            }
            return series.map(self.label_maps[feature_name])
        else:
            # 測試階段：使用現有映射，將未見過的類別映射到特殊值
            mapping = self.label_maps[feature_name]
            max_val = max(mapping.values())
            return series.map(lambda x: mapping.get(x, max_val + 1))

    def convert_time_to_minutes(self, time_str):
        """將時間字符串轉換為分鐘數"""
        try:
            hours, minutes = map(int, time_str.split(":"))
            return hours * 60 + minutes
        except:
            print(f"無法解析時間格式: {time_str}")
            return 0  # 返回默認值而不是 None

    def preprocess_data(self, df, is_training=True):
        # 創建數據的副本
        data = df.copy()

        # 處理類別型特徵
        categorical_features = ["att3", "att6", "att7", "att8", "att9"]

        for feature in categorical_features:
            data[feature] = self.safe_label_encode(data[feature], feature, is_training)

        # 處理時間特徵
        data["minutes_from_midnight"] = data["att1"].apply(self.convert_time_to_minutes)
        data["hour"] = data["minutes_from_midnight"] // 60

        # 創建時間段特徵
        data["time_period"] = pd.cut(
            data["hour"],
            bins=[-1, 5, 11, 16, 21, 24],
            labels=["late_night", "morning", "afternoon", "evening", "night"],
        )
        data["time_period"] = self.safe_label_encode(
            data["time_period"], "time_period", is_training
        )

        # 創建週期性時間特徵
        minutes_in_day = 24 * 60
        data["time_sin"] = np.sin(
            2 * np.pi * data["minutes_from_midnight"] / minutes_in_day
        )
        data["time_cos"] = np.cos(
            2 * np.pi * data["minutes_from_midnight"] / minutes_in_day
        )

        # 計算地理距離
        data["distance"] = np.sqrt(
            (data["att12"] - data["att15"]) ** 2 + (data["att13"] - data["att16"]) ** 2
        )

        # 特徵組合
        data["amount_per_distance"] = data["att4"] / (
            data["distance"] + 1
        )  # 避免除以零
        data["amount_time_factor"] = data["att4"] * np.abs(
            data["time_sin"]
        )  # 交易金額和時間的關係

        # 選擇要使用的特徵
        features = [
            "att4",  # 交易金額
            "att5",  # 持卡人年齡
            "hour",  # 小時
            "time_period",  # 時間段
            "time_sin",  # 週期性時間特徵（正弦）
            "time_cos",  # 週期性時間特徵（餘弦）
            "distance",  # 地理距離
            "att10",  # 城市人口
            "att3",  # 交易類別
            "att6",  # 性別
            "att7",  # 職業
            "att8",  # 城市
            "att9",  # 州
            "amount_per_distance",  # 單位距離的交易金額
            "amount_time_factor",  # 交易金額時間因子
        ]

        X = data[features]

        # 標準化數值特徵
        numerical_features = [
            "att4",
            "att5",
            "distance",
            "att10",
            "time_sin",
            "time_cos",
            "amount_per_distance",
            "amount_time_factor",
        ]
        if is_training:
            self.scaler.fit(X[numerical_features])
        X[numerical_features] = self.scaler.transform(X[numerical_features])

        return X

    def train(self, train_data):
        print("開始訓練模型...")

        # 預處理訓練數據
        X = self.preprocess_data(train_data, is_training=True)
        y = train_data["fraud"]

        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 計算類別權重
        total = len(y_train)
        fraud_ratio = sum(y_train) / total
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        # 訓練模型
        eval_set = [(X_val, y_val)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        # 進行交叉驗證
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"\n交叉驗證分數: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 在驗證集上評估模型
        y_pred = self.model.predict(X_val)
        print("\n分類報告:")
        print(classification_report(y_val, y_pred))

        print("\n混淆矩陣:")
        print(confusion_matrix(y_val, y_pred))

        # 輸出特徵重要性
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": self.model.feature_importances_}
        )
        print("\n特徵重要性:")
        print(feature_importance.sort_values("importance", ascending=False))

    def predict(self, test_data):
        # 預處理測試數據
        X_test = self.preprocess_data(test_data, is_training=False)

        # 進行預測
        predictions = self.model.predict(X_test)

        # 創建提交文件
        submission = pd.DataFrame({"Id": test_data["Id"], "fraud": predictions})

        return submission


# 使用示例
if __name__ == "__main__":
    # 讀取數據
    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/test_data.csv")

    # 創建和訓練模型
    model = FraudDetectionModel()
    model.train(train_data)

    # 進行預測並保存結果
    predictions = model.predict(test_data)
    predictions.to_csv("submission.csv", index=False)
    print("\n預測結果已保存到 'submission.csv'")
