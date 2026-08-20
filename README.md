# Indo TTS Runner

一個簡單的獨立腳本，把印尼文文字轉成語音（TTS），用的是社群微調的 [alkhrzmy/qwen3-tts-0.6b-indonesian](https://huggingface.co/alkhrzmy/qwen3-tts-0.6b-indonesian) 模型（在 Qwen3-TTS-0.6B 上針對印尼語微調）。

**注意跟中文那個 QwenTTS 工具不一樣的地方**：

- 這個模型只有一種固定音色（`indonesian_speaker`），不能像 QwenTTS 那樣選其他預設語者、用文字設計聲音，或複製（clone）特定人的聲音。
- 這是**社群微調模型**，不是 Qwen 官方釋出的（官方 Qwen3-TTS 目前不支援印尼語）。訓練設定相對簡單（4 epochs），品質沒有官方保證，實際使用前建議自己多聽幾段確認效果可以接受。

**這個資料夾不含模型檔案**——模型權重會在第一次執行時自動從 Hugging Face 下載（快取在 `~/.cache/huggingface`），完全不用手動下載。

## 安裝步驟

跟中文版 QwenTTS 用同一個 conda 環境即可（如果同事電腦上還沒有，就照下面步驟建立）：

1. 建立一個 Python 環境：

   ```bash
   conda create -n qwen3-tts python=3.12
   conda activate qwen3-tts
   ```

2. 安裝 PyTorch。如果有 NVIDIA 顯卡，去 [PyTorch 官網安裝頁](https://pytorch.org/get-started/locally/) 找對應的 CUDA 版本安裝，例如：

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   沒有顯卡就裝 CPU 版本（這個 0.6B 模型不大，CPU 也能跑，只是比 GPU 慢）：

   ```bash
   pip install torch torchaudio
   ```

3. 安裝 Qwen-TTS 套件：

   ```bash
   pip install qwen-tts soundfile
   ```

## 使用方法

把想要念出來的印尼文文字放進腳本旁邊的 `text.txt`，然後執行：

```bash
python run_indo_tts.py
```

或在 Windows 上直接雙擊 `run_indo_tts.bat`（如果你的 conda 沒裝在預設路徑，記得先打開這個檔案改一下裡面的 `CONDA_BASE`）。

也可以直接用命令列指定文字：

```bash
python run_indo_tts.py --text "Halo, apa kabar?"

# 用其他檔案當輸入
python run_indo_tts.py --text_file naskah.txt

# 讓每個句號「.」後面停頓 0.4 秒
python run_indo_tts.py --period_pause 0.4
```

輸出的 wav 檔會存在 `indo_tts_output/` 資料夾裡。
