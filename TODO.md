# 多模態即時情緒辨識 — 待辦清單 (TODO)

## 專案目標
- 4 類情緒辨識：**快樂 (happy) / 憤怒 (angry) / 悲傷 (sad) / 平靜 (neutral)**
- 模態：**影像（人臉身分 + 表情）+ 音訊（語音情緒）**，不含文字
- 實作方式：預訓練模型抽特徵 + 訓練 4 類輕量頭（sklearn）
- 環境：Python + **僅 CPU**

---

## 里程碑 1：骨架與環境  ← 進行中
- [x] 建立專案目錄結構
- [x] 撰寫 requirements.txt
- [x] 撰寫 configs/config.yaml
- [x] 實作鏡頭擷取模組 `capture/video_source.py`
- [x] 實作麥克風擷取模組 `capture/audio_source.py`
- [x] 撰寫 app/main.py 最小可跑範例
- [x] 建立 venv (.venv) 並安裝核心套件 (numpy/opencv/sounddevice/PyYAML/scikit-learn)
- [ ] 安裝系統音訊函式庫 libportaudio2 (需 sudo，使用者執行)
- [x] 安裝其餘套件 (onnxruntime/onnx；表情辨識已改用 ONNX 版 HSEmotion，免 torch)
- [ ] 驗證鏡頭與錄音可正常運作

## 里程碑 2：人臉身分
- [x] SCRFD 偵測 + 5 點對齊 (face_recognition/detector.py, align.py)
- [x] ArcFace/MobileFaceNet embedder (face_recognition/embedder.py)
- [x] SQLite gallery 註冊與比對 (face_recognition/gallery.py)
- [x] CLI 註冊工具 (face_recognition/register.py)
- [x] 管線整合 (face_recognition/pipeline.py)
- [x] ONNX 模型下載腳本 (scripts/download_models.py)
- [x] 影像測試通過 (scripts/test_face.py: 6 臉、512 維、自身比對正確)
- [ ] 即時身分比對與 FPS 量測 (app/face_demo.py，需在目標裝置執行)

## 里程碑 3：特徵抽取管線
- [x] HSEmotion 人臉表情特徵抽取 (face_emotion/extractor.py；ONNX，1280 維特徵 + 4 類機率)
- [ ] emotion2vec+ 語音特徵抽取
- [ ] 下載並篩選中文 4 類資料集 (RAF-DB, MAFW, CASIA, ESD, CSEMOTIONS)
- [x] 台灣華語資料集調研 (見 docs/datasets_tw.md)
- [ ] 送出 BIIC-Podcast / TFEID 申請 (審核耗時，先送不佔工)
- [ ] 評估免申請方案：zero-shot (emotion2vec+/HSEmotion) 或免費語料 (CSEMOTIONS/ESD/EmotionTalk)
- [ ] 決策：語音資料取得方式（免費語料 / 自行錄製台灣小標註集 / 爬蟲擴增）
- [ ] 特徵存成 .npy (training/extract_features.py)

## 里程碑 4：訓練 4 類頭
- [ ] 語音 4 類頭訓練 (sklearn, 交叉驗證)
- [ ] 人臉 4 類頭訓練 (sklearn, 交叉驗證)
- [ ] 依說話者/身分切分 train/test
- [ ] 輸出 4x4 混淆矩陣與指標

## 里程碑 5：即時推論接線
- [ ] 人臉表情即時 4 類輸出
- [ ] 語音情緒即時 4 類輸出
- [ ] 畫面疊字顯示 (app/overlay.py)

## 里程碑 6：多模態融合
- [ ] 時間戳對齊 (fusion/aligner.py)
- [ ] 決策融合 (fusion/decision_fusion.py)
- [ ] 特徵融合 (fusion/feature_fusion.py)
- [ ] 訓練融合頭並比較
- [ ] **模態不對稱風險**：語音準確度可能明顯低於臉部 (見 docs/datasets_tw.md §8)；先量測落差
- [ ] 機率校準 (temperature scaling) 後再融合
- [ ] 信心加權 / 學習式 fusion (取代單純平均或 concat)
- [ ] graceful degradation：語音低信心時退回只信臉部
- [ ] 評估分開報：臉部 / 語音 / 融合三者各自 metrics

## 里程碑 7：評估
- [ ] 單模態 vs 融合 Acc / macro-F1
- [ ] 端到端延遲 / FPS / CPU 使用率
- [ ] JSON/CSV 事件記錄

## 目前問題 / 待決策 (Issues)
- [ ] **硬體**：WSL 無鏡頭與麥克風 → 即時擷取/推論功能延後到目標裝置驗證
- [ ] **台灣標註資料集全需申請**：BIIC-Podcast、TFEID 等無免申請公開版本；申請需時
- [ ] **免申請替代**：路線 A 用預訓練模型 zero-shot（不需資料集）；路線 B 用免費對岸語料
      (CSEMOTIONS、ESD、EmotionTalk 影音多模態、ASVP-ESD) + 自行錄製台灣小標註集
- [ ] **語音/臉部準確度不對稱**：跨語料/跨腔調 SER 可能明顯低於臉部 (見 docs/datasets_tw.md §8)
- [ ] **爬蟲可行性**：可抓「未標註」台灣語音/影音做擴增，但**無法取得台灣標註**；
      標籤須靠弱監督/偽標籤/人工標註；有著作權、平台 ToS、個資法、IRB 風險
- [ ] **決策未定**：語音資料改採 免費語料 / 自行錄製 / 爬蟲擴增（待選）
- [ ] **授權釐清**：免申請語料多為簡中且多數 CC BY-NC（限非商業研究）

## 環境注意
- 執行環境為 WSL Ubuntu 22.04 + Python 3.11（`.venv`）
- 麥克風需系統函式庫 PortAudio：`sudo apt-get install -y libportaudio2 portaudio19-dev`
- WSL 的鏡頭/麥克風需要 Windows 端支援：
  - 麥克風：WSLg 會透過 PulseAudio 轉發（Windows 11 通常可用）
  - 網路攝影機：需 `usbipd-win` 將 USB 鏡頭掛載進 WSL，或改用 Windows 端 Python 執行

## 備註
- 非商業授權資料集僅供研究使用
- 台灣專屬標註資料集多需 EULA/申請；不可用爬蟲繞過授權
- 大型權重與資料集不進版控 (見 .gitignore)
