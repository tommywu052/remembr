# ReMEmbR Wheeled-Legged Robot Demo

輪足機器人 ReMEmbR 記憶建構與推理系統。已成功整合至 xiaozhi Realtime API，支援中文語音查詢記憶並自動導航。

## 架構

```
┌─────────────────────────────────────────────────┐
│                Jetson Orin NX 16GB                │
│                                                  │
│  RealSense → CaptionerNode ──HTTP──┐             │
│  Nav2/AMCL → MemoryBuilderNode     │             │
│            → Milvus Lite (.db) ◄───│─── xiaozhi  │
│                                    │      │      │
│  xiaozhi (Realtime API) ──────────│──────┼──────┼── Azure Cloud
│   ├ query_robot_memory → LiteMemory│      │      │
│   ├ navigate_leggedrobot → Nav2    │      │      │
│   └ 中文語音 ↔ GPT ↔ 工具呼叫     │      │      │
└────────────────────────────────────│──────┘      │
                                     │              │
                      ┌──────────────▼──────────────┘
                      │  Remote RTX PRO 6000
                      │  Ollama (192.168.31.63)
                      │  ├── gemma4:e4b (VLM+LLM+翻譯)
                      │  └── nomic-embed-text (Embedding)
                      └─────────────────────────────
```

### 與原版 ReMEmbR (Nova Carter) 的差異

| 項目 | 原版（Nova Carter） | 本版本（輪足） |
|------|---------------------|---------------|
| VLM（影像描述） | 本地 VILA 3B + NanoLLM Docker | Remote Gemma4:e4b（更強，Jetson 零負擔） |
| LLM（推理） | 本地 Ollama command-r | Remote Gemma4:e4b + tool calling |
| Embedding | 本地 sentence-transformers (1024-dim) | Remote nomic-embed-text (768-dim) |
| 向量 DB | Docker MilvusDB Standalone | Milvus Lite（本地檔案，無需 Docker） |
| ASR | 本地 Whisper TRT | xiaozhi Realtime API (Azure OpenAI) |
| 語音整合 | ROS2 /speech topic | xiaozhi 直接呼叫 LiteMemory + navigate |
| Jetson 本地依賴 | PyTorch + VILA + Ollama + Docker | 僅 pymilvus + requests |

## 檔案結構

| 檔案 | 用途 |
|------|------|
| `ollama_client.py` | Ollama REST API 輕量客戶端（caption / chat / embed / tool calling） |
| `lite_memory.py` | Milvus Lite 記憶後端（含位置去重搜尋、圖片路徑存儲） |
| `captioner_node.py` | ROS2 節點：RealSense 影像 → Remote Gemma4 caption → `/caption`（同步存圖） |
| `memory_builder_node.py` | ROS2 節點：`/caption` + `/amcl_pose` → Milvus Lite（解析圖片路徑） |
| `agent_node.py` | ROS2 節點：`/speech` → Remote Gemma4 推理 → `/goal_pose`（獨立使用） |
| `memory_viewer.py` | 記憶檢視器：生成 HTML 報告（位置地圖 + 圖片 + caption 對照表） |
| `test_pipeline.py` | 端到端測試（不需 ROS2） |
| `README.md` | 本文件 |

### ROS2 Topic 關係

```
/camera/color/image_raw (sensor_msgs/Image)
        │
        ▼
  CaptionerNode ─── HTTP ──→ Ollama gemma4:e4b (VLM caption)
        │
        ▼
  /caption (std_msgs/String)
        │
        ├──────────────────┐
        ▼                  ▼
  /amcl_pose          MemoryBuilderNode ─── HTTP ──→ Ollama nomic-embed-text
  (PoseWithCovariance       │
   Stamped)                 ▼
        │            Milvus Lite DB
        │            (remembr_memory.db)
        │                  │
        ▼                  ▼
  /speech ──────→ AgentNode ─── HTTP ──→ Ollama gemma4:e4b (LLM reasoning)
  (std_msgs/        │
   String)          ▼
             /goal_pose (PoseStamped) ──→ Nav2 自動導航
```

## 快速測試（不需 ROS2）

```bash
cd ~/legged_robot/remembr/examples/wheeled_legged_demo
python3 test_pipeline.py
```

驗證項目：Ollama 連線 → Embedding → Milvus Lite → 記憶寫入 → 搜尋 → Agent 推理。

## 完整啟動流程

### Phase 1：基礎節點

```bash
# Terminal 1: 底盤驅動（必須最先啟動）
ros2 run wheeled_legged_pkg wl_base_node \
  --ros-args -p serial_port:=/dev/robot_base -p auto_select:=true

# Terminal 2: RealSense 攝影機
ros2 launch realsense2_camera rs_launch.py \
  config_file:=~/legged_robot/realsense/realsense_params.yaml
```

### Phase 2：導航模式（提供 /amcl_pose 定位，需已建圖）

```bash
# Terminal 3: LiDAR
cd ~/legged_robot/LSLIDAR_X_ROS2/src && source install/setup.bash
ros2 launch lslidar_driver lsn10_launch.py

# Terminal 4: Nav2 完整導航棧（含 AMCL 定位 + RViz 視覺化）
source ~/legged_robot/ROS2_Packages/install/setup.bash
ros2 launch wheeled_legged_pkg navigation_with_lidar.launch.py

# Terminal 5: 設定初始位姿（機器人開機位置對應地圖原點）
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  '{header: {frame_id: "map"}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]}}'
```

> 或者用一鍵啟動腳本：
> ```bash
> cd ~/legged_robot/ROS2_Packages/src/wheeled_legged_pkg/
> ./scripts/start_full_navigation.sh
> ```

### Phase 3：ReMEmbR 記憶建構

```bash
# Terminal 6: Captioner（影像 → Remote Gemma4 VLM → caption 文字描述）
cd ~/legged_robot/remembr/examples/wheeled_legged_demo
python3 captioner_node.py

# Terminal 7: Memory Builder（caption + pose → Milvus Lite 向量資料庫）
python3 memory_builder_node.py
```

### Phase 3.5：巡邏建構記憶

記憶建構需要機器人在環境中移動，讓 CaptionerNode 拍攝影像並產生描述。
`memory_builder_node` 訂閱 `/amcl_pose`（由 Nav2 AMCL 提供），因此**導航模式必須啟動**。

AMCL 是被動定位，不管機器人用什麼方式移動，它都會持續追蹤位置。以下操控方式**都可以使用**：

| 操控方式 | 推薦度 | 說明 |
|---------|--------|------|
| **HC 搖控器** | **推薦** | 最直覺、最自由。SBUS 訊號走硬體通道，與 Nav2 不衝突 |
| **RViz Nav Goal** | 可用 | 在 RViz 地圖上點「2D Nav Goal」，Nav2 自動規劃路徑 |
| **xiaozhi 語音** | 可用 | 說「前進」「轉左」或 `navigate_leggedrobot` 導航指令 |
| **ros2 topic pub /cmd_vel** | 可用 | 手動發 Twist 命令，操作較不方便 |

**巡邏技巧：**

- **速度放慢** — Caption 間隔預設 5 秒，太快移動會漏拍場景
- **各方向都看** — 到達一個區域時原地轉一圈，讓攝影機看到四周環境
- **留意 RViz** — 確認 AMCL 粒子雲收斂在正確位置（定位正常）
- **觀察 Terminal 7** — Memory Builder 的 log 會顯示每次記憶寫入，確認 pipeline 正常
- **覆蓋完整** — 走過每個房間、走廊、重要區域，記憶越豐富查詢結果越準確

### Phase 4：記憶查詢與導航

記憶建構完成後，**停止 Phase 3 的 captioner 和 memory_builder**（釋放 DB 鎖），然後：

#### 方案 A：透過 xiaozhi 語音查詢（推薦）

```bash
# Terminal 6: 啟動 xiaozhi（自動載入 remembr_memory.db）
cd ~/xiaozhi
python3.10 -u py-xiaozhi-realtime-rtc-robotctl.py
# 瀏覽器開啟 http://localhost:8088
# 直接用中文語音：「帶我去廚房」「走廊在哪」「你看到什麼」
```

#### 方案 B：透過 agent_node + ros2 topic（獨立測試用）

```bash
# Terminal 6: Agent（接收問題 → 記憶檢索 → LLM 推理 → 發布導航目標）
cd ~/legged_robot/remembr/examples/wheeled_legged_demo
python3.10 agent_node.py

# Terminal 7: 測試查詢
ros2 topic pub --once /speech std_msgs/String "data: Hey robot, where can I find snacks?"
ros2 topic pub --once /speech std_msgs/String "data: Hey robot, take me to the sofa"
```

> agent_node 注意：預設過濾關鍵字為 "robot"，查詢內容需包含 "robot" 才會觸發。
> 可透過 `--ros-args -p query_keyword:=""` 關閉過濾。

### Phase 5：記憶檢視與驗證

巡邏完成後，可以生成 HTML 報告來檢視所有記憶，包含 2D 位置地圖、原始圖片和 caption 對照：

```bash
# 先停止佔用 DB 的程式（captioner / memory_builder / xiaozhi）
cd ~/legged_robot/remembr/examples/wheeled_legged_demo

# 生成報告並啟動 HTTP 伺服器
python3.10 memory_viewer.py --serve
# 瀏覽器開啟 http://<Jetson-IP>:8099/memory_report.html
```

報告內容：
- **統計卡片** — 總記憶數、唯一位置數、座標範圍、有圖片的記憶數
- **2D 位置地圖** — 所有記憶點的空間分佈，顏色表示時間先後
- **位置聚類表** — 哪些位置記憶最多（可發現停留過久的區域）
- **完整記憶表格** — 每條記憶的圖片、時間、座標、caption，支援搜尋與圖片放大

> 圖片功能需要使用新版 captioner_node（`save_images=true`），舊的記憶沒有圖片但仍可查看 caption 和位置。
> 重新巡邏前需刪除舊 DB（`rm remembr_memory.db`）以啟用新 schema 的 `image_path` 欄位。

## ROS2 參數一覽

### CaptionerNode

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `ollama_host` | `192.168.31.63` | Remote Ollama 主機 IP |
| `ollama_port` | `11434` | Ollama API 埠 |
| `vlm_model` | `gemma4:e4b` | 用於影像描述的 VLM 模型 |
| `image_topic` | `/camera/camera/color/image_raw` | RealSense 影像 topic |
| `caption_topic` | `/caption` | 輸出的 caption topic |
| `caption_interval` | `5.0` | Caption 產生間隔（秒） |
| `save_images` | `true` | 是否保存每次 caption 對應的圖片 |
| `image_save_dir` | `./memory_images` | 圖片保存目錄 |

### MemoryBuilderNode

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `db_path` | `./remembr_memory.db` | Milvus Lite 資料庫檔案路徑 |
| `db_collection` | `robot_memory` | 向量集合名稱 |
| `ollama_host` | `192.168.31.63` | Remote Ollama（用於 embedding） |
| `pose_topic` | `/amcl_pose` | 機器人定位 topic |
| `caption_topic` | `/caption` | 訂閱的 caption topic |

### AgentNode

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `llm_model` | `gemma4:e4b` | 用於推理的 LLM 模型 |
| `db_path` | `./remembr_memory.db` | Milvus Lite 資料庫（需與 MemoryBuilder 相同） |
| `db_collection` | `robot_memory` | 向量集合名稱（需與 MemoryBuilder 相同） |
| `query_topic` | `/speech` | 接收查詢的 topic |
| `pose_topic` | `/amcl_pose` | 機器人目前位姿（加入查詢上下文） |
| `goal_pose_topic` | `/goal_pose` | 發布導航目標的 topic |
| `query_keyword` | `robot` | 查詢過濾關鍵字（空字串 = 不過濾） |

## 與 xiaozhi Realtime API 整合（已完成）

已整合至 `~/xiaozhi/py-xiaozhi-realtime-rtc-robotctl.py`，新增 `query_robot_memory` 工具。
使用者可直接用**中文語音**詢問記憶位置並導航前往。

### 整合架構

```
使用者（中文語音）
    │
    ▼
Azure OpenAI Realtime API (GPT)
    │
    ├─ query_robot_memory(query="kitchen")  ← GPT 自動翻譯成英文
    │       │
    │       ▼
    │   LiteMemory.search_by_text()
    │       │  ↕ HTTP → Ollama nomic-embed-text
    │       ▼
    │   記憶結果：「廚房在 (4.86, 4.16)」
    │
    ├─ navigate_leggedrobot(x=4.86, y=4.16)
    │       │
    │       ▼
    │   Nav2 → 機器人自動導航
    │
    └─ 語音回覆：「好的，我記得廚房在那邊，正在帶你過去！」
```

### 關鍵設計

| 設計決策 | 說明 |
|----------|------|
| **英文 Caption** | VLM (`gemma4:e4b`) 以英文描述場景，配合 `nomic-embed-text` 英文 embedding 模型 |
| **查詢自動翻譯** | GPT 被提示用英文 query；若仍收到中文，系統自動透過 Ollama 翻譯後再搜尋 |
| **位置去重** | `search_by_text` 先取 `limit×6` 候選，再以 0.5m 半徑去重，避免起始點大量記憶淹沒結果 |
| **ROS2 Context 自動恢復** | `get_ros_node()` 每次使用前驗證 context 有效性，失效則自動重建 |
| **圖片保存** | CaptionerNode 同步存圖到 `memory_images/`，搭配 `memory_viewer.py` 可視覺驗證 caption 品質 |

### 已驗證的語音查詢範例

| 中文語音 | GPT 查詢 | 記憶位置 | 導航結果 |
|----------|----------|----------|----------|
| 「帶我去廚房」 | `kitchen` | `(4.86, 4.16)` | 成功 |
| 「帶我到架子附近」| `shelf` | `(-0.66, -0.04)` | 成功 |
| 「你有看到長廊嗎」| `hallway` | `(-0.61, 0.00)` | 成功 |
| 「門框在哪」 | `door frame` | `(4.65, 0.69)` | 成功 |
| 「掛畫在哪裡」 | `painting on wall` | `(6.77, 2.10)` | 成功 |

### 使用方式

不需要啟動 `agent_node.py`（已被 xiaozhi 取代），只需：

```bash
# 1. 啟動 ROS2 基礎 + 導航（Phase 1-2）
# 2. 啟動記憶建構（Phase 3），巡邏完成後停止 captioner / memory_builder
# 3. 啟動 xiaozhi（會自動載入 remembr_memory.db）
cd ~/xiaozhi
python3.10 -u py-xiaozhi-realtime-rtc-robotctl.py
```

啟動時會顯示：
```
  Memory:     啟用 (135 條記憶)
```

> **注意**：Milvus Lite 不支援多進程同時開啟同一個 DB 檔案。
> 啟動 xiaozhi 前請先停止 `memory_builder_node` 和 `agent_node`。

## 已知問題與解法

| 問題 | 原因 | 解法 |
|------|------|------|
| 所有查詢回傳同一位置 | 起始點記憶過多 + 語意搜尋被多數票淹沒 | `search_by_text` 加入 0.5m 半徑位置去重 |
| 中文查詢找不到正確記憶 | `nomic-embed-text` 不支援跨語言匹配 | GPT 提示用英文 query + 自動翻譯層（Ollama gemma4） |
| Milvus Lite 多進程鎖定 | SQLite 不支援多 writer | 依序操作：先建構記憶，停止後再啟動查詢 |
| ROS2 context invalid | 長時間運行後 rclpy context 失效 | `get_ros_node()` 自動偵測並重建 context / publisher |
| Python 版本衝突 | conda Python 3.13 vs ROS2 Python 3.10 | 明確使用 `python3.10` 執行所有 ROS2 相關程式 |
| numpy 版本衝突 | numpy 2.x 與 pandas/opencv 不相容 | 鎖定 `numpy<2.0`（使用 1.26.4） |

## 系統需求

| 項目 | 需求 |
|------|------|
| Jetson | Orin NX 16GB（已驗證） |
| Python | **必須使用 python3.10**（ROS2 Humble 綁定） |
| Python 套件 | `pymilvus>=2.4`, `milvus-lite`, `requests`, `numpy<2.0` |
| Remote | Ollama + gemma4:e4b + nomic-embed-text（RTX PRO 6000 已驗證） |
| ROS2 | Humble（含 Nav2, realsense2_camera, lslidar_driver） |
| 地圖 | 需事先用 SLAM 建圖完成 |
