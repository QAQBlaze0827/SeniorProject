# SeniorProject

多模態即時情緒辨識系統（4 類：快樂 / 憤怒 / 悲傷 / 平靜）。

- 模態：影像（人臉身分 + 表情）+ 音訊（語音情緒）
- 執行環境：Python + CPU
- 方法：預訓練模型特徵 + 訓練 4 類輕量分類頭

## 環境安裝

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

在 WSL 環境則使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 執行（里程碑 1：鏡頭 + 錄音測試）

```
python -m app.main
```

按 `q` 關閉視窗。

## 執行（里程碑 1：鏡頭 + 錄音測試）

```
python -m app.main
```

按 `q` 關閉視窗。

## 人臉辨識（里程碑 2）

先下載 ONNX 模型（SCRFD 偵測 + MobileFaceNet 辨識 + HSEmotion 表情）：

```
python scripts/download_models.py
```

驗證管線（對單張影像偵測 + 抽特徵 + 自身比對）：

```
python scripts/test_face.py tests/data/t1.jpg
```

註冊身分（用資料夾內影像，一個資料夾一個人）：

```
python -m face_recognition.register --name Alice --folder path/to/Alice
```

註冊身分（用鏡頭即時，按空白鍵擷取、q 離開）：

```
python -m face_recognition.register --name Alice --camera --count 5
```

即時人臉辨識（需鏡頭）：

```
python -m app.face_demo
```

模型與資料集如需在另一台裝置執行，請在該裝置重新執行 `scripts/download_models.py`。

## 人臉表情（里程碑 3）

以 HSEmotion (EfficientNet-B0) ONNX 模型抽取表情特徵，輸出 8 類機率再映射為 4 類。

驗證（對單張影像偵測人臉 + 抽 1280 維特徵 + 4 類機率）：

```
python scripts/test_face_emotion.py tests/data/t1.jpg
```

## 文件

- [TODO.md](TODO.md)：里程碑與待辦
- [docs/datasets_tw.md](docs/datasets_tw.md)：台灣華語情緒資料集清單，含「語音/臉部準確度不對稱」的風險與緩解設計

## 進度
待辦與里程碑請見 [TODO.md](TODO.md)。
