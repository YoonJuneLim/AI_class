# 📂 Colab에서 CSV 파일 가져오기

> Google Drive에 저장된 CSV 파일을 Google Colab에서 불러오는 방법

---

## 🗂️ 사전 준비

- [ ] Google Drive에 CSV 파일 업로드 완료
- [ ] Google Colab 접속

---

## 📋 단계별 가이드

### Step 1 — Google Drive에 CSV 파일 업로드

Google Drive에 사용할 CSV 파일을 미리 업로드해 둔다.

---

### Step 2 — Google Colab 실행

[colab.research.google.com](https://colab.research.google.com) 에 접속하여 새 노트북을 열거나 기존 노트북을 실행한다.

---

### Step 3 — 좌측 파일 패널 열기

왼쪽 사이드바에서 📁 **파일** 아이콘을 클릭한다.

---

### Step 4 — 드라이브 마운트 클릭

파일 패널 상단의 **"드라이브 마운트 해제"** (Google Drive 아이콘) 버튼을 클릭한다.

> 💡 이미 마운트된 경우에는 해제 후 재마운트하거나, 마운트된 상태를 그대로 사용하면 된다.

---

### Step 5 — 권한 할당 및 Drive 연결

팝업 창이 뜨면 Google 계정에 대한 **권한을 허용**하여 Drive를 Colab에 연결한다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### Step 6 — Drive 폴더 생성 확인

파일 패널에서 `/content/drive/MyDrive` 폴더가 생성된 것을 확인한다.  
**MyDrive** 를 클릭하면 업로드한 파일 목록이 보인다.

---

### Step 7 — 파일 경로 복사

원하는 CSV 파일에서 **우클릭 → "경로 복사"** 를 선택한다.

복사된 경로 예시:
```
/content/drive/MyDrive/data/sample.csv
```

---

### Step 8 — 코드에 경로 붙여넣기

복사한 경로를 코드에 붙여넣어 파일을 불러온다.

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/data/sample.csv')
df.head()
```

---

## ✅ 완료

이제 CSV 파일을 DataFrame으로 불러와 분석을 시작할 수 있다! 🎉