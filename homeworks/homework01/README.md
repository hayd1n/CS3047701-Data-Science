English | [中文](#信用卡詐欺檢測模型說明與報告)

# Credit Card Fraud Detection Model Report

> **NTUST Course Project**
>  Course No: `CS3047701`
>  Course Name: Introduction to Data Science
>  Author: Hayden Chang (張皓鈞) B11030202
>  Email: [B11030202@mail.ntust.edu.tw](mailto:B11030202@mail.ntust.edu.tw)

## Core Framework of the Model Design

### Overall Workflow

1. **Data Preprocessing**: Clean raw data and perform multi-level feature engineering to transform it into a model-ready format while extracting latent useful features.
2. **Model Training**: Employ an XGBoost algorithm, leveraging class weight adjustments to address class imbalance.
3. **Model Evaluation**: Use classification reports, confusion matrices, and cross-validation to measure the model’s performance and ensure its ability to detect fraud samples.
4. **Prediction**: Apply the trained model to test data and generate specific prediction results.

### Key Design Features

- **Modular Structure**:
   Functions are broken down into independent methods (e.g., `preprocess_data`, `train`, `predict`), enhancing debugging and reusability.
- **Scalability**:
   Supports the addition of new features, encoding schemes, and alternative classification algorithms.
- **Adaptation to Class Imbalance**:
   Automatically calculates weights for positive and negative samples based on the fraud sample ratio, improving sensitivity to minority classes.
- **High Interpretability**:
   Provides feature importance analysis to understand the key factors influencing model predictions.

## Data Processing and Feature Engineering

### Temporal Feature Processing

- **Time Transformation**: Convert raw time (e.g., "12:30") into minutes since midnight, extracting hour and cyclic features (sine and cosine).
- **Time Period Segmentation**: Classify transactions by time periods (e.g., late night, morning, afternoon) to analyze behavioral patterns.

### Categorical Feature Encoding

- Safe Label Encoding:
  - Training phase: Generate unique numeric mappings for categorical features.
  - Testing phase: Encode unseen categories as a special value to avoid errors.

### Geographic and Distance Features

- **Geographic Distance**: Calculate Euclidean distance between two geographic coordinates (e.g., transaction location and cardholder location) to analyze anomalies.
- **Combination of Distance and Amount**:
  - Compute the ratio of transaction amount to geographic distance to detect abnormal behaviors of "per-distance transaction amount."
  - Introduce an "amount-time factor" to combine amount and cyclic time features, exploring time-sensitive transaction patterns.

### Feature Standardization

- Numerical features are standardized to ensure numerical stability and faster convergence, transforming them into a normal distribution.

## Model Training and Evaluation

### Training Design

- **XGBoost Algorithm**:
  - A gradient-boosting tree model suitable for nonlinear features and anomalies.
  - Parameters (e.g., tree depth, learning rate, column sampling rate) are optimized for better performance.
- **Class Weight Adjustment**:
  - Dynamically set the `scale_pos_weight` parameter based on the fraud sample ratio, enhancing sensitivity to fraud detection.

### Evaluation Metrics

- **Cross-Validation**: 5-fold cross-validation ensures stability across different data splits.
- **Classification Report**: Outputs precision, recall, and F1-score for each class.
- **Confusion Matrix**: Provides counts for TP, FP, TN, and FN, aiding analysis of fraud detection.
- **Feature Importance**: Ranks features by their impact on model decisions, facilitating business insights.

## Prediction and Output Results

On the test data, the model performs the following steps:

1. **Preprocessing**: Apply the same feature engineering techniques used in training.
2. **Prediction**: Generate fraud probability labels for each transaction.
3. **Save Results**: Export predictions in CSV format for further analysis or submission.

## Running the program

This model is written in [Python](https://www.python.org/), make sure you have Python installed.

> This program was tested in Python 3.10 only.

### Installation dependencies

```bash
pip install -r requirements.txt
```

### Run the main program

```bash
python main.py
```

Predictions are saved to ``submission.csv``.

## Results Analysis and Explanation

```
Cross-validation scores: 0.9937 (+/- 0.0014)

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7884
           1       0.71      0.78      0.74       116

    accuracy                           0.99      8000
   macro avg       0.85      0.89      0.87      8000
weighted avg       0.99      0.99      0.99      8000


Confusion Matrix:
[[7846   38]
 [  25   91]]

Feature Importance:
                feature  importance
0                  att4    0.331531
3           time_period    0.204558
13  amount_per_distance    0.118589
8                  att3    0.099216
5              time_cos    0.048145
11                 att8    0.030960
4              time_sin    0.026598
14   amount_time_factor    0.024074
2                  hour    0.023757
1                  att5    0.022275
10                 att7    0.021152
7                 att10    0.017389
6              distance    0.013373
12                 att9    0.011890
9                  att6    0.006494
```

### Cross-Validation Scores

- **Result**: `0.9937 (+/- 0.0014)`

  - The mean cross-validation score is **0.9937**, with a standard deviation of **0.0014**.
  - **Interpretation**: The model demonstrates high accuracy (~99.37%) and minimal performance variance, reflecting strong adaptability to data features.

  **Analysis**:

  - This result indicates excellent learning of data patterns by the model.
  - However, the high overall accuracy may partly stem from strong performance on non-fraud samples, potentially overlooking challenges in fraud detection.

### Classification Report

- **Precision**:
  - Fraud samples (`1`): **0.71**.
  - Non-fraud samples (`0`): **1.00**.
  - **Interpretation**: 71% of transactions predicted as fraud are true fraud cases, while non-fraud samples are rarely misclassified.
- **Recall**:
  - Fraud samples: **0.78**.
  - Non-fraud samples: **1.00**.
  - **Interpretation**: The model detects 78% of fraud samples but misses 22%.
- **F1-Score**:
  - Fraud samples: **0.74**. A balance of precision and recall.
- **Macro Average** and **Weighted Average**:
  - Macro Avg: Reflects balanced performance across both classes (F1-score: **0.87**).
  - Weighted Avg: Heavily influenced by non-fraud samples, showing overall performance (F1-score: **0.99**).

### Confusion Matrix

- **Result**:

  ```
  [[7846   38]
   [  25   91]]
  ```

  - Non-fraud samples:
    - **TP**: 7846 correctly predicted as non-fraud.
    - **FP**: 38 misclassified as fraud (~0.48% error rate).
  - Fraud samples:
    - **TP**: 91 correctly detected (~78.4% detection rate).
    - **FN**: 25 missed fraud cases (~21.6% undetected).

  **Interpretation**:

  - The model performs exceptionally well on non-fraud samples but shows room for improvement in detecting fraud cases.

### Feature Importance

| Rank | Feature                       | Importance (%) |
| ---- | ----------------------------- | -------------- |
| 1    | `att4` (Transaction Amount)   | 33.15%         |
| 2    | `time_period` (Time Period)   | 20.46%         |
| 3    | `amount_per_distance`         | 11.86%         |
| 4    | `att3` (Transaction Category) | 9.92%          |
| 5    | `time_cos` (Cosine of Time)   | 4.81%          |

**Key Observations**:

1. **Transaction Amount (`att4`)**: The most significant predictor; high-value transactions are more likely to be flagged as fraud.
2. **Time Period (`time_period`)**: Behavioral patterns during certain periods (e.g., late night) influence predictions.
3. **Geographic-Amount Features (`amount_per_distance`)**: Captures the risk of high-value transactions over long distances.
4. **Other Features**: Time cyclicity and transaction categories provide additional context but are less impactful.

## Model Strengths and Challenges

### Strengths

1. **High Accuracy**: Excellent performance on non-fraud samples.
2. **Interpretability**: Feature importance offers actionable business insights.
3. **Stability**: Consistent cross-validation scores ensure reliability.

### Challenges

1. **Lower Fraud Precision**: Some non-fraud transactions are misclassified as fraud, impacting user experience.
2. **Missed Fraud Cases**: 25 undetected fraud transactions may pose risks.
3. **Class Imbalance**: Limited fraud samples hinder effective learning for minority classes.

### Conclusion

The model delivers outstanding results on non-fraud transactions but has room for improvement in fraud detection, especially in reducing false positives and enhancing sensitivity to fraud cases.



# 信用卡詐欺檢測模型說明與報告

## 模型設計的核心架構

### 整體流程

1. **數據預處理**：清洗數據並進行多層次的特徵工程，將原始數據轉化為模型可接受的數字格式，同時提取潛在的有用特徵。
2. **模型訓練**：使用基於決策樹的 XGBoost 演算法，結合類別權重調整來應對類別不平衡的問題。
3. **模型評估**：採用分類報告、混淆矩陣及交叉驗證來衡量模型的表現，確保其對詐欺樣本的檢測能力。
4. **模型預測**：將訓練後的模型應用於測試數據，生成具體的預測結果。

### 主要設計特點

- 模組化結構：

  - 各種功能分解為獨立方法（如 `preprocess_data`、`train`、`predict` 等），便於調試和重用。

- 可擴展性：

  - 支援新增特徵、不同類別編碼方案及其他分類演算法。

- 適配類別不平衡問題：

  - 根據詐欺樣本的比例，自動計算正負樣本的權重，提升對少數類別的敏感性。

- 解釋性強：

  - 提供特徵重要性分析，幫助了解哪些特徵對模型判斷影響最大。

## 數據處理與特徵工程

### 時間特徵處理

- **時間轉換**：將原始時間（如 "12:30"）轉換為距午夜的分鐘數，進一步提取小時及週期性特徵（正弦和餘弦）。
- **時間段劃分**：根據一天的不同時段（如深夜、早晨、下午等）對交易進行分類，加入行為模式分析。

### 類別特徵編碼

- 使用「安全標籤編碼」：
  - 訓練階段：為類別特徵生成唯一的數字映射。
  - 測試階段：將未見過的類別編碼為特殊值，避免模型出錯。

### 地理與距離特徵

- **地理距離**：根據兩組地理座標（如交易地和持卡人所在地）計算歐幾里得距離，用於分析異常交易的地理分佈。

- 距離與金額特徵組合：

  - 計算交易金額與地理距離的比值，揭示「單位距離交易金額」的異常行為。
- 引入「金額時間因子」，將金額與週期性時間進行結合，探索交易金額的時間敏感性。

### 特徵標準化

- 為確保模型的數值穩定性與收斂速度，對數值型特徵進行標準化處理，使其符合常態分佈。

## 模型訓練與評估

### 訓練設計

- 模型使用 **XGBoost** 演算法：
  - 它是一種基於梯度增強的樹模型，對非線性特徵和異常數據具有良好的適應性。
  - 參數設置靈活，例如樹深度、學習率和列抽樣率等均經過調優。
- **類別權重調整**：
  - 計算詐欺樣本的比例，動態設置 `scale_pos_weight` 參數，讓模型對於不平衡數據中的少數類別（詐欺）更加敏感。

### 模型評估

- **交叉驗證**：通過 5 折交叉驗證評估模型在不同數據劃分上的穩定性。
- **分類報告**：輸出每個類別的精確率（Precision）、召回率（Recall）和 F1 分數。
- **混淆矩陣**：提供 TP、FP、TN、FN 的具體數據，幫助分析模型對詐欺樣本的檢測能力。
- **特徵重要性**：排序各特徵對模型決策的影響，有助於業務分析。

## 預測與結果輸出

在測試數據上，模型進行以下操作：

1. 預處理：應用與訓練數據相同的特徵處理方法。
2. 預測：生成每筆交易的詐欺可能性標籤。
3. 保存結果：以 CSV 格式輸出，供進一步分析或提交。

## 運行程式

本模型使用 [Python](https://www.python.org/) 編寫，請確保你已安裝 Python。

> 本程式僅在 Python 3.10 中測試

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 執行主程式

```bash
python main.py
```

預測結果將保存到 `submission.csv`

## 結果分析與解釋

```
Cross-validation scores: 0.9937 (+/- 0.0014)

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      7884
           1       0.71      0.78      0.74       116

    accuracy                           0.99      8000
   macro avg       0.85      0.89      0.87      8000
weighted avg       0.99      0.99      0.99      8000


Confusion Matrix:
[[7846   38]
 [  25   91]]

Feature Importance:
                feature  importance
0                  att4    0.331531
3           time_period    0.204558
13  amount_per_distance    0.118589
8                  att3    0.099216
5              time_cos    0.048145
11                 att8    0.030960
4              time_sin    0.026598
14   amount_time_factor    0.024074
2                  hour    0.023757
1                  att5    0.022275
10                 att7    0.021152
7                 att10    0.017389
6              distance    0.013373
12                 att9    0.011890
9                  att6    0.006494
```


### 交叉驗證分數
- **結果**：`0.9937 (+/- 0.0014)`
  - 平均交叉驗證分數為 **0.9937**，標準差為 **0.0014**。
  - **解釋**：模型在不同數據切分上的表現非常穩定，整體準確率極高（接近 99.37%）。標準差小，說明模型的表現波動很小。
  - **分析**：
    - 此結果表明模型能很好地學習數據特徵，適應性強。
    - 不過，由於詐欺樣本比例極低，高整體準確率可能部分來自於對非詐欺樣本的優秀預測，而未充分反映詐欺樣本檢測的困難。

### 分類報告
- **Precision (精確率)**：
  - 詐欺樣本（類別 `1`）的精確率為 **0.71**。
  - 非詐欺樣本（類別 `0`）的精確率為 **1.00**。
  - **解釋**：
    - 對於預測為詐欺的交易，71% 是正確的（真陽性）。
    - 對於非詐欺樣本，模型幾乎不會誤判為詐欺。
- **Recall (召回率)**：
  - 詐欺樣本的召回率為 **0.78**，非詐欺樣本為 **1.00**。
  - **解釋**：
    - 模型成功檢測了 78% 的詐欺樣本（真陽性），但仍有 22% 的詐欺交易未檢測出來（假陰性）。
- **F1-score**：
  - 詐欺樣本的 F1 分數為 **0.74**，是一個平衡精確率和召回率的綜合指標。
- **宏平均 (Macro avg)** 和 **加權平均 (Weighted avg)**：
  - 宏平均：該指標對每個類別給予相等權重，適合類別不平衡情況，顯示模型在兩類別之間的整體平衡性（F1-score: **0.87**）。
  - 加權平均：按樣本數量加權，受非詐欺樣本的影響較大，反映整體性能（F1-score: **0.99**）。

**總結**：
- 模型在非詐欺樣本上的表現非常出色，但在詐欺樣本上的檢測能力還有提升空間，尤其是精確率（0.71）可能導致誤報一些非詐欺交易為詐欺。

### 混淆矩陣
- **結果**：
  ```
  [[7846   38]   # 非詐欺樣本：7846 正確預測為非詐欺；38 誤報為詐欺。
   [  25   91]]  # 詐欺樣本：91 正確檢測；25 未能檢測出來。
  ```
- **解釋**：
  - 非詐欺樣本：
    - 錯誤率非常低（38/7884 ≈ 0.48%），幾乎所有非詐欺交易都被正確標記。
  - 詐欺樣本：
    - 偵測率較高（91/116 ≈ 78.4%），但仍有 25 筆詐欺交易未能識別（假陰性）。
    - **平衡挑戰**：模型在保證非詐欺交易正確率的同時，詐欺交易的精確率（71%）顯示存在一定的誤報。

### 特徵重要性
- **結果分析**：
  | 排名 | 特徵名稱             | 重要性 (%) |
  |------|----------------------|------------|
  | 1    | `att4`（交易金額）   | 33.15%     |
  | 2    | `time_period`（時間段） | 20.46%     |
  | 3    | `amount_per_distance` | 11.86%     |
  | 4    | `att3`（交易類別）   | 9.92%      |
  | 5    | `time_cos`（時間餘弦） | 4.81%      |

- **關鍵觀察**：
  1. **交易金額 (`att4`) 是最重要的特徵**：
     - 高額交易可能更容易被判定為潛在詐欺。
  2. **時間段 (`time_period`) 是第二重要特徵**：
     - 不同時段的交易行為模式（如深夜交易）可能更易引發警報。
  3. **地理與金額相關特徵（`amount_per_distance`）**：
     - 地理距離與交易金額的結合是第三大影響因子，反映了異地高金額交易的潛在風險。
  4. **交易類別 (`att3`) 和時間週期性特徵（`time_cos`）**：
     - 交易的行為模式（如類別和時間規律）也提供了重要的判斷依據。
  5. 其他特徵（如性別 `att6` 和州 `att9`）權重較低，對結果的貢獻有限。

### 模型優勢與挑戰
#### 優勢
1. **高整體準確率**：模型在非詐欺樣本上的正確率極高，對大部分交易能準確分類。
2. **特徵解釋性強**：特徵重要性提供了清晰的業務指導。
3. **穩定性好**：交叉驗證分數表現穩定，結果具有可靠性。

#### 挑戰
1. **詐欺樣本的精確率偏低**：
   - 部分非詐欺交易被誤判為詐欺，可能對用戶體驗產生影響。
2. **未檢測出的詐欺交易**：
   - 模型未能檢測的 25 筆詐欺交易可能導致實際風險。
3. **類別不平衡問題**：
   - 詐欺樣本數量遠低於非詐欺樣本，模型可能對少數類別的學習不充分。

### 結論
該模型對非詐欺交易的預測效果極佳，但在詐欺檢測上仍有改進空間。
