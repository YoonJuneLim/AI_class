# ML / DL 코드 메뉴얼 (복사 붙여넣기용)

## 변경해야 할 것만 확인
- `[피처수]` : X의 컬럼 수 → `d_c_x.shape[1]` 로 확인
- `[클래스수]` : classification 클래스 수 (2개면 binary, 3개 이상이면 multi-class)
- `[타겟컬럼]` : y로 사용할 컬럼명 (예: 'target', 'bmi')

---

## STEP 1. 데이터 로드

### 방법 1 — sklearn 내장 데이터셋
```python
import pandas as pd
import numpy as np
from sklearn import datasets

data = datasets.load_wine()       # load_breast_cancer, load_diabetes, load_iris 등
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.columns)
```

### 방법 2 — CSV 파일 (index 컬럼 있을 때)
```python
import pandas as pd
import numpy as np

df = pd.read_csv('파일명.csv', index_col=0)

print(df.head())
print(df.columns)
```

### 방법 3 — CSV 파일 (index 컬럼 없을 때)
```python
import pandas as pd
import numpy as np

df = pd.read_csv('파일명.csv')

print(df.head())
print(df.columns)
```

---

## STEP 2. EDA

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 기본 정보 확인
print(df.shape)           # 행(샘플 수), 열(피처 수)
print(df.dtypes)          # 각 컬럼 데이터 타입
print(df.describe())      # 평균, 표준편차, 최솟값, 최댓값 등 통계 요약

# 레이블 분포 확인 (classification용)
sns.countplot(data=df, x='target')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.show()

print(df['target'].value_counts())  # 클래스별 샘플 수

# 결측치 확인
print(df.isnull().sum())            # 컬럼별 결측치 수
print(df.isnull().sum().sum())      # 전체 결측치 수

# 각 컬럼 분포 확인 (히스토그램)
fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
df.hist(ax=ax)
plt.show()

# 박스플롯 — 이상치 확인
plt.figure(figsize=(15, 6))
df.boxplot()
plt.xticks(rotation=45)
plt.show()

# 타겟과 피처 간 산점도 (regression용)
plt.scatter(df['피처명'], df['target'])
plt.xlabel('피처명')
plt.ylabel('target')
plt.show()
```

---

## STEP 3. 결측치 제거

```python
df = df.dropna()
print(df.isnull().sum())
```

---

## STEP 4. Feature Selection

### (1) Random Forest
```python
# classification 데이터
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)

# regression 데이터 (위 줄과 교체)
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(random_state=0)

rf.fit(df.drop('target', axis=1), df['target'])

importances = pd.Series(rf.feature_importances_, index=df.drop('target', axis=1).columns)
importances.sort_values(ascending=False).head(10)
```

### (2) Correlation Heatmap
```python
import seaborn as sns
import matplotlib.pyplot as plt

cf_matrix = df.corr()
plt.figure(figsize=(15, 15))
sns.set(font_scale=0.6)
sns.heatmap(cf_matrix, annot=True, cbar=False, fmt='.1f')
plt.show()

corr_target = df.corr()['target'].abs().sort_values(ascending=False)
print(corr_target.head(10))
```

### (3) Pairplot
```python
sns.pairplot(df,
             vars=['컬럼1', '컬럼2', '컬럼3', '컬럼4'],  # 상위 4개 피처
             hue='target',
             palette='Set1',
             diag_kind='kde')
plt.show()
```

---

## STEP 5. 데이터 분리

```python
# classification
c_df = df.copy()
c_y = c_df['target']
c_x = c_df.drop('target', axis=1)

# regression (y가 target이 아닌 경우 [타겟컬럼] 변경)
r_df = df.copy()
r_y = r_df['[타겟컬럼]']
r_x = r_df.drop(['[타겟컬럼]'], axis=1)
```

---

## STEP 6. ML 모델

### Classification
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(c_x, c_y, test_size=0.2, random_state=0)

c1, c2, c3 = LogisticRegression(random_state=0), DecisionTreeClassifier(random_state=0), RandomForestClassifier(random_state=0)
c1.fit(X_train, y_train); c2.fit(X_train, y_train); c3.fit(X_train, y_train)
c1_y, c2_y, c3_y = c1.predict(X_test), c2.predict(X_test), c3.predict(X_test)

print(f'Logistic Regression \n {classification_report(y_test, c1_y)}')
print(f'Decision Tree \n {classification_report(y_test, c2_y)}')
print(f'Random Forest \n {classification_report(y_test, c3_y)}')
```

### Regression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(r_x, r_y, test_size=0.2, random_state=0)

r1, r2, r3 = LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()
r1.fit(X_train, y_train); r2.fit(X_train, y_train); r3.fit(X_train, y_train)
r1_y, r2_y, r3_y = r1.predict(X_test), r2.predict(X_test), r3.predict(X_test)

print(f'Linear Regression \n {mean_squared_error(y_test, r1_y)}')
print(f'Decision Tree \n {mean_squared_error(y_test, r2_y)}')
print(f'Random Forest \n {mean_squared_error(y_test, r3_y)}')
```

---

## STEP 7. DL 데이터 준비

### 정규화 방법 비교

| 방법 | 범위 | 사용 시기 |
|---|---|---|
| `StandardScaler` | 평균 0, 표준편차 1 (범위 제한 없음) | 이상치가 있을 때, 일반적인 경우 |
| `MinMaxScaler` | 0 ~ 1 | 이상치 없고 범위를 0~1로 맞추고 싶을 때 |
| `MinMaxScaler((-1, 1))` | -1 ~ 1 | sigmoid/tanh 활성화 함수 사용 시, 음수 데이터 포함 시 |

> **-1~1 선택 이유:** sigmoid(0~1), tanh(-1~1) 같은 활성화 함수의 입력 범위와 맞춰주면 학습이 더 안정적. 음수 데이터가 있을 때 0~1로 강제하면 음수 정보가 손실되므로 -1~1 사용.

### 정규화 O (권장 — 피처 범위 차이가 클 때)
```python
from sklearn.preprocessing import StandardScaler

# classification
scaler_c = StandardScaler()
d_c_x = scaler_c.fit_transform(c_x)
d_c_y = pd.get_dummies(c_y).values   # one-hot encoding

# regression
scaler_x = StandardScaler()
scaler_y = StandardScaler()
d_r_x = scaler_x.fit_transform(r_x)
d_r_y = scaler_y.fit_transform(r_y.values.reshape(-1, 1)).flatten()

print('classification input shape:', d_c_x.shape[1])
print('regression input shape:', d_r_x.shape[1])
```

### 정규화 O — MinMaxScaler (0~1 또는 -1~1)
```python
from sklearn.preprocessing import MinMaxScaler

# 0~1 범위
scaler = MinMaxScaler()
d_c_x = scaler.fit_transform(c_x)

# -1~1 범위
scaler = MinMaxScaler(feature_range=(-1, 1))
d_c_x = scaler.fit_transform(c_x)
```

### 정규화 X (피처 범위가 비슷할 때)
```python
# classification
d_c_x = c_x.values
d_c_y = pd.get_dummies(c_y).values   # one-hot encoding

# regression
d_r_x = r_x.values
d_r_y = r_y.values

print('classification input shape:', d_c_x.shape[1])
print('regression input shape:', d_r_x.shape[1])
```

> **정규화 기준:**
> - 피처 간 범위 차이가 크면 → 정규화 O
> - 이미 정규화된 sklearn 데이터셋 → 정규화 선택적
> - DL에서 학습이 안 될 때 → 정규화 추가

---

## STEP 8. DL 모델 - Classification

### Binary (클래스 2개)
```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential  # 층을 순서대로 쌓는 모델
from keras.layers import Dense        # 완전연결층 (모든 노드가 연결됨)
from keras.optimizers import Adam     # 학습 최적화 알고리즘

X_train, X_test, y_train, y_test = train_test_split(d_c_x, d_c_y, test_size=0.2, random_state=0)

model = Sequential()
# Dense(노드수, input_shape=입력크기, activation=활성화함수)
# 노드수: 많을수록 복잡한 패턴 학습 가능, 너무 많으면 과적합
# relu: 음수는 0, 양수는 그대로 → 은닉층에서 주로 사용
model.add(Dense(10, input_shape=([피처수],), activation='relu'))  # 입력층
model.add(Dense(8, activation='relu'))                             # 은닉층
model.add(Dense(6, activation='relu'))                             # 은닉층
model.add(Dense(1, activation='sigmoid'))  # 출력층: binary는 1개, sigmoid → 0~1 확률 출력

# Adam: 학습률을 자동 조절하는 optimizer (가장 일반적으로 사용)
# learning_rate: 학습 속도, 너무 크면 발산, 너무 작으면 느림
# binary_crossentropy: 2클래스 분류 손실함수
model.compile(Adam(learning_rate=0.001), 'binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### Multi-class (클래스 3개 이상)
```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train, X_test, y_train, y_test = train_test_split(d_c_x, d_c_y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(10, input_shape=([피처수],), activation='relu'))  # 입력층
model.add(Dense(8, activation='relu'))                             # 은닉층
model.add(Dense(6, activation='relu'))                             # 은닉층
# softmax: 각 클래스의 확률 합이 1이 되도록 출력 → multi-class에서 사용
model.add(Dense([클래스수], activation='softmax'))  # 출력층: 클래스 수만큼 노드

# categorical_crossentropy: one-hot 인코딩된 다중 클래스 손실함수
model.compile(Adam(learning_rate=0.001), 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## STEP 9. DL 모델 - Regression

```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train, X_test, y_train, y_test = train_test_split(d_r_x, d_r_y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(10, input_shape=([피처수],), activation='relu'))  # 입력층
model.add(Dense(8, activation='relu'))                             # 은닉층
model.add(Dense(6, activation='relu'))                             # 은닉층
# 출력층: activation 없음 → 연속값 그대로 출력 (regression은 범위 제한 없어야 함)
model.add(Dense(1))

# mse(Mean Squared Error): regression 손실함수, 예측값과 실제값의 차이 제곱 평균
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
model.summary()
```

---

## STEP 10. 학습

```python
model_history = model.fit(
    x=X_train, y=y_train,        # 학습 데이터
    epochs=100,                  # 전체 데이터를 몇 번 반복 학습할지 (많을수록 정확하지만 과적합 위험)
    batch_size=32,               # 한 번에 학습할 샘플 수 (작을수록 정확하지만 느림)
    validation_data=(X_test, y_test))  # 매 epoch마다 테스트 데이터로 성능 모니터링

y_pred = model.predict(X_test)   # 학습된 모델로 예측
```

> **epochs 기준:** 학습 loss는 내려가는데 val_loss가 올라가기 시작하면 과적합 → epochs 줄이기
> **batch_size 기준:** 일반적으로 32 또는 64 사용

### 학습 곡선 확인 (선택)
```python
import matplotlib.pyplot as plt

# loss 곡선
plt.plot(model_history.history['loss'], label='Train loss')
plt.plot(model_history.history['val_loss'], label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# accuracy 곡선 (classification만)
plt.plot(model_history.history['accuracy'], label='Train accuracy')
plt.plot(model_history.history['val_accuracy'], label='Val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## STEP 11. 평가

### Classification 평가
```python
from sklearn.metrics import classification_report, confusion_matrix

# binary (정규화 없이 sigmoid 사용 시)
y_pred_class = (y_pred > 0.5).astype(int).flatten()
y_test_class = y_test

# multi-class (one-hot 사용 시)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))
```

### Regression 평가
```python
from sklearn.metrics import mean_squared_error, r2_score

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f'MSE:  {mse}')
print(f'RMSE: {rmse}')
print(f'R²:   {r2}')
```

---

## 핵심 정리표

| 상황 | 출력층 | 손실함수 | y 형태 | 예측 변환 |
|---|---|---|---|---|
| Binary classification | `Dense(1, sigmoid)` | `binary_crossentropy` | one-hot | `> 0.5` |
| Multi-class classification | `Dense(N, softmax)` | `categorical_crossentropy` | one-hot | `argmax` |
| Regression | `Dense(1)` activation 없음 | `mse` | 연속값 | 그대로 |

---

## 정확도 판단 기준

### Classification 지표
| 지표 | 좋음 | 나쁨 | 설명 |
|---|---|---|---|
| accuracy | 1.0에 가까울수록 | 0에 가까울수록 | 전체 정답률 |
| precision | 1.0에 가까울수록 | 0에 가까울수록 | 예측한 것 중 실제 정답 비율 |
| recall | 1.0에 가까울수록 | 0에 가까울수록 | 실제 정답 중 맞춘 비율 |
| f1-score | 1.0에 가까울수록 | 0에 가까울수록 | precision + recall 종합 |

> **일반 기준:** 0.9 이상 → 좋음 / 0.7~0.9 → 보통 / 0.7 미만 → 나쁨

### Regression 지표
| 지표 | 좋음 | 나쁨 | 설명 |
|---|---|---|---|
| MSE | 0에 가까울수록 | 클수록 | 오차의 제곱 평균 (단위 해석 어려움) |
| RMSE | 0에 가까울수록 | 클수록 | MSE의 제곱근 (실제 단위와 동일) |
| R² | 1.0에 가까울수록 | 음수면 최악 | 데이터를 얼마나 설명하는지 비율 |

> **R² 기준:** 0.9 이상 → 좋음 / 0.7~0.9 → 보통 / 0.5 미만 → 나쁨 / 음수 → 학습 실패

---

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| accuracy가 에포크 내내 고정 | 정규화 없음 또는 손실함수 오류 | StandardScaler 추가 또는 손실함수 확인 |
| R² 음수 | 학습 실패 | learning_rate 조정 또는 epochs 증가 |
| 모델이 한 클래스만 예측 | softmax + binary_crossentropy 조합 오류 | 손실함수를 categorical_crossentropy로 변경 |
| ValueError: input shape | input_shape 불일치 | `d_c_x.shape[1]` 로 확인 후 수정 |
