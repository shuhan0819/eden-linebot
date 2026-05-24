# 伊甸聊天機器人 LINE Bot

一個部署於 LINE 平台的 RAG（檢索增強生成）健康諮詢聊天機器人，以 Breeze-7B 繁體中文語言模型為核心，結合 Google Drive 文件知識庫，提供溫暖、口語化的健康資訊查詢服務。

---

## 功能特色

- **RAG 知識庫查詢**：自動從 Google Drive 下載文件（PDF、Word、Markdown、JSON），建立 FAISS 向量資料庫，回答使用者的健康相關問題
- **繁體中文語言模型**：使用 Ollama 在地端執行 `jcai/breeze-7b-instruct-v1_0:q4_0`，全程繁體中文回覆
- **語音輸入與輸出**：支援語音訊息輸入（Whisper 語音辨識）與 TTS 語音合成輸出（Edge TTS，可切換男女聲）
- **意圖分類**：自動辨別閒聊、情緒支持、健康提問，分流至不同回覆策略
- **偽查詢擴充檢索**：生成假設性答案片段輔助 FAISS 向量搜尋，提升檢索精確度
- **Fallback 機制**：知識庫無相關資料時自動改用模型自身常識回答，並記錄至 Google Sheets
- **五星評分系統**：對話結束時透過 Flex Message 收集使用者評分，寫入 Google Sheets
- **對話歷史管理**：保留每位使用者最近 5 輪對話，支援追問與代名詞解析

---

## 系統需求

| 項目 | 版本 / 說明 |
|------|------------|
| Python | 3.10 以上 |
| Ollama | 需在本機執行，並已拉取 `jcai/breeze-7b-instruct-v1_0:q4_0` |
| ngrok | 用於將本機服務暴露為公開 HTTPS URL，供 LINE Webhook 使用 |
| FFmpeg | pydub 音訊處理所需（需安裝並加入 PATH） |
| LINE Developers 帳號 | 需建立 Messaging API Channel |
| Google Cloud 服務帳號 | 需啟用 Google Drive API 與 Google Sheets API |

---

## 安裝步驟

### 1. 建立並啟動虛擬環境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 安裝並啟動 Ollama，拉取語言模型

```bash
# 安裝 Ollama 後，拉取 Breeze 模型
ollama pull jcai/breeze-7b-instruct-v1_0:q4_0

# 在背景啟動 Ollama 服務
nohup ollama serve > ollama.log 2>&1 &
```

### 4. 設定環境變數

```bash
# 複製範本
cp .env.example .env

# 編輯 .env，填入所有實際金鑰與 ID
```

### 5. 放置 Google 服務帳號金鑰

將 Google Cloud Console 下載的服務帳號 JSON 金鑰檔案放至專案根目錄，命名為 `service_account.json`。

---

## 環境變數說明

請參照 `.env.example`，複製為 `.env` 後逐一填寫：

| 變數名稱 | 說明 | 取得方式 |
|---------|------|---------|
| `LINE_CHANNEL_SECRET` | LINE Channel 驗證密鑰 | LINE Developers Console > Basic settings |
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Messaging API 存取金鑰 | LINE Developers Console > Messaging API |
| `NGROK_URL` | ngrok 公開 HTTPS URL（例如 `https://xxxx.ngrok-free.app`） | 啟動 ngrok 後取得 |
| `GOOGLE_DRIVE_FOLDER_ID` | 知識庫文件所在的 Google Drive 資料夾 ID | 從資料夾分享網址中擷取 |
| `GOOGLE_SPREADSHEET_ID` | 記錄評分與查無資料問題的 Google Spreadsheet ID | 從試算表網址中擷取 |
| `SERVICE_ACCOUNT_FILE` | 服務帳號 JSON 金鑰路徑（預設 `service_account.json`） | Google Cloud Console |

---

## 使用方式

### 啟動服務

```bash
python eden_linebot.py
```

服務預設監聽 `http://0.0.0.0:8000`。

### 設定 LINE Webhook

1. 啟動 ngrok：`ngrok http 8000`
2. 將取得的 HTTPS URL 填入 `.env` 的 `NGROK_URL`
3. 至 LINE Developers Console 設定 Webhook URL：`https://your-ngrok-url/callback`
4. 開啟 **Use webhook**

### LINE Bot 特殊指令

| 指令 | 說明 |
|------|------|
| `！重置小伊的知識庫！` | 重新從 Google Drive 下載文件並重建向量資料庫 |
| `！結束使用！` | 結束對話並顯示五星評分介面 |
| `！切換男聲！` / `！切換女聲！` | 切換 TTS 語音合成聲音 |

---

## 專案結構

```
伊甸聊天機器人/
├── eden_linebot.py   # 主程式（FastAPI + LINE Bot + RAG 全流程）
├── service_account.json    # Google 服務帳號金鑰（請勿上傳，已列入 .gitignore）
├── .env                    # 環境變數（請勿上傳，已列入 .gitignore）
├── .env.example            # 環境變數範本（可上傳）
├── .gitignore              # Git 忽略清單
├── requirements.txt        # Python 套件相依清單
├── README.md               # 本說明文件
│
├── audio/                  # 執行時自動建立，儲存語音檔案（.aac / .mp3 / .m4a）
├── drive_documents/        # 執行時自動建立，存放從 Google Drive 下載的文件
└── breeze_vectorstores/    # 執行時自動建立，存放 FAISS 向量資料庫
```

---

## 注意事項與已知限制

- **語言模型需在本機運行**：本專案使用 Ollama 在地端推論，不依賴雲端 API，因此對硬體有一定需求。使用 q4_0 量化版本，建議至少 8GB RAM。
- **首次啟動較慢**：啟動時會自動從 Google Drive 下載文件並建立向量資料庫，依文件數量需等待數分鐘。向量資料庫建立後會快取於磁碟，後續啟動速度較快。
- **ngrok 免費版 URL 每次重啟會變更**：每次重啟 ngrok 後需更新 `.env` 中的 `NGROK_URL` 並重新設定 LINE Webhook URL。
- **對話歷史僅保存於記憶體**：伺服器重啟後，所有使用者的對話歷史將清空。
- **ffmpeg 為必要依賴**：若系統未安裝 ffmpeg 或未加入 PATH，語音功能將無法正常運作。
- **Ollama 同時僅允許一個推論**：為避免 CPU 資源競爭，系統使用 Semaphore 確保同一時間只有一個 Ollama 呼叫在執行，並行請求會排隊等待。
