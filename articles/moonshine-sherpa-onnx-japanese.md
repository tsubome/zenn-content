---
title: "Moonshine Voice ASR を sherpa-onnx で動かして日本語文字起こしする完全ガイド"
emoji: "🎙"
type: "tech"
topics: ["moonshine", "whisper", "asr", "sherpaonnx", "python"]
published: false
---

## はじめに

**Moonshine Voice** は、Google TensorFlow チーム創設メンバーが立ち上げた Moonshine AI（旧 Useful Sensors）が開発するオープンソースの音声認識（ASR）モデルです。Hacker News で 316 ポイントを獲得するなど、今まさに注目を集めています。

その注目の理由は **「Whisper Large v3 と同等精度を、パラメータ数 1/6 で実現」** という驚異的な効率。さらにエッジデバイス（スマホ、Raspberry Pi）でリアルタイム動作することを前提に設計されています。

「じゃあ日本語はどうなの？」

この記事では、Moonshine の日本語モデルを [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) で実際に動かし、**11 分〜72 分の講義音声で精度を検証**した結果を共有します。VAD パラメータの最適化では **2,040 パターン以上のグリッドサーチ** を行い、F1 スコアを 54% → 96.8% まで改善しました。

## Moonshine のモデル世代を正しく理解する

Moonshine には **3 つの世代** があり、これを混同すると正確な議論ができません。

### 世代1: Original Moonshine（2024年10月）

- 論文: [arXiv:2410.15608](https://arxiv.org/abs/2410.15608)
- **英語のみ**、Tiny (27.1M params) / Base (61.5M params)
- Full-attention encoder + RoPE
- Whisper と違い **ゼロパディングなし** → 音声長に比例した計算量

### 世代1.5: Flavors of Moonshine（2025年9月）

- 論文: [arXiv:2509.02523](https://arxiv.org/abs/2509.02523)
- **日本語を含む 6 言語の単言語特化モデル**
- アーキテクチャは v1 と同じ（full-attention encoder）
- 同サイズの多言語モデルより大幅に高精度
- Tiny (27.1M) に加え、Base (61.5M) も後日追加

### 世代2: Moonshine Streaming（2026年2月）

- 論文: [arXiv:2602.12241](https://arxiv.org/abs/2602.12241)
- **全く新しいアーキテクチャ**: sliding-window self-attention
- **英語のみ**（2026年3月時点）
- Tiny Streaming (34M) / Small Streaming (123M) / Medium Streaming (245M)

### 公式モデル一覧

[GitHub README](https://github.com/moonshine-ai/moonshine#available-models) の Available Models テーブルより:

| 言語 | アーキテクチャ | パラメータ | WER/CER |
|------|--------------|-----------|---------|
| English | Tiny | 27.1M | 12.66% |
| English | Tiny Streaming | 33.6M | 12.00% |
| English | Base | 61.5M | 10.07% |
| English | Small Streaming | 123M | 7.84% |
| English | Medium Streaming | 245M | 6.65% |
| Arabic | Base | 61.5M | 5.63% |
| **Japanese** | **Base** | **61.5M** | **13.62%** |
| Korean | Tiny | 27.1M | 6.46% |
| 他4言語 | Base | 61.5M | — |

**日本語モデルは Flavors 世代（v1 アーキテクチャ）** です。v2 Streaming ではありません。

:::message alert
**sherpa-onnx の `from_moonshine_v2()` は紛らわしい命名です。** これは Moonshine v2（Streaming）とは無関係で、ONNX エクスポート形式の世代（4ファイル → 2ファイル）を指します。英語の Tiny/Base も同じ `from_moonshine_v2()` で使えることがその証拠です。
:::

## 環境構築

### インストール

```bash
pip install sherpa-onnx numpy
```

### モデルのダウンロード

HuggingFace から sherpa-onnx 用の量子化モデルを取得します。

```bash
# Base JA (135MB, 高精度)
git lfs install
git clone https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-base-ja-quantized-2026-02-27

# Tiny JA (69MB, 高速)
git clone https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27
```

Silero VAD モデルも必要です:

```bash
wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

### 最小動作確認

```python
import sherpa_onnx

recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
    encoder="moonshine-base-ja/encoder_model.ort",
    decoder="moonshine-base-ja/decoder_model_merged.ort",
    tokens="moonshine-base-ja/tokens.txt",
    num_threads=4,
)

stream = recognizer.create_stream()
# 16kHz float32 の音声データを渡す
stream.accept_waveform(16000, audio_samples)
recognizer.decode_stream(stream)
print(stream.result.text)
```

これだけで動きます。しかし実用上は **VAD（Voice Activity Detection）との組み合わせが必須** です。

## VAD + Moonshine パイプライン

### なぜ VAD が必要か

Moonshine（Flavors 世代）には **入力長の上限が約 9 秒** という制約があります。これを超えると ONNX ランタイムが以下のエラーを返します:

```
Attempting to broadcast an axis by a dimension other than 1. 2 by N
```

そのため、長い音声を適切な長さに分割する VAD が不可欠です。

### パイプライン構成

```
音声 (16kHz WAV)
  │
  ▼
VoiceActivityDetector (Silero VAD)
  - 512 samples (32ms) ずつ投入
  - 発話区間を自動検出・分割
  │
  ▼
OfflineRecognizer (Moonshine)
  - セグメントごとに文字起こし
  - 9秒超のセグメントはハードカット + オーバーラップ
  │
  ▼
後処理
  - CJK 文字間スペース除去
  - ファジーマージ（オーバーラップ部分の重複除去）
```

### 基本コード

```python
import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000

# --- VAD 作成 ---
def create_vad(model_path):
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model_path
    config.silero_vad.threshold = 0.12
    config.silero_vad.min_silence_duration = 1.2
    config.silero_vad.min_speech_duration = 0.3
    config.silero_vad.max_speech_duration = 7.0
    config.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=300)

# --- VAD 実行 ---
def run_vad(audio, vad_model_path):
    vad = create_vad(vad_model_path)
    window = 512
    offset = 0
    while offset + window <= len(audio):
        vad.accept_waveform(audio[offset:offset + window])
        offset += window
    # 残りをゼロパディングして投入
    if offset < len(audio):
        pad = np.zeros(window, dtype=np.float32)
        pad[:len(audio) - offset] = audio[offset:]
        vad.accept_waveform(pad)
    vad.flush()  # ← 最後のセグメントを確定（重要！）

    segments = []
    while not vad.empty():
        seg = vad.front
        segments.append((np.array(seg.samples, dtype=np.float32), seg.start))
        vad.pop()
    return segments

# --- 文字起こし ---
segments = run_vad(audio, "silero_vad.onnx")
for samples, start in segments:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    print(f"@{start/SAMPLE_RATE:.1f}s: {stream.result.text}")
```

:::message
**`vad.flush()` を忘れないでください。** これを呼ばないと、最後の発話区間が確定せず消失します。
:::

## 落とし穴: VoiceActivityDetector vs VadModel

sherpa-onnx には VAD の API が **2 種類** あります:

| API | セグメント管理 | パラメータの効果 |
|-----|-------------|---------------|
| `VoiceActivityDetector` | **自動** | min_silence 等が正しく機能 |
| `VadModel.is_speech()` | **手動** | min_silence 等が効かない |

`VadModel.is_speech()` はフレーム単位の raw 判定で、内部のステートトラッキングが働きません。実際に試した結果:

| API | セグメント数 | F1 |
|-----|-----------|-----|
| `VadModel.is_speech()` | 221 (平均 0.8秒) | **54%** |
| `VoiceActivityDetector` | 95 (平均 5.3秒) | **96.8%** |

**必ず `VoiceActivityDetector` を使ってください。**

## VAD パラメータチューニング

### Moonshine の入力長制約

| サンプル数 | 秒数 | 結果 |
|-----------|------|------|
| 128,000 | 8.0s | OK |
| 144,000 | 9.0s | OK（ギリギリ） |
| 158,464 | 9.9s | **ONNX エラー** |

最短側にも制約があり、約 0.1 秒未満ではエラーになります。安全な範囲は **0.15 秒〜8.5 秒** です。

### 4 つのパラメータ

| パラメータ | 説明 | デフォルト | 最適値 |
|-----------|------|---------|--------|
| `threshold` | 音声/非音声の判定閾値 | 0.5 | **0.12** |
| `min_silence_duration` | この秒数の無音でセグメント確定 | 0.25 | **1.2** |
| `min_speech_duration` | この秒数未満の発話を無視 | 0.25 | **0.3** |
| `max_speech_duration` | セグメントの最大長 | 5.0 | **7.0** |

### max_speech_duration が最も重要

4 軸の中で **`max_speech_duration` が F1 に最も大きな影響** を与えます:

```
max=5.0:  F1=95.3%  ← セグメントが短すぎて文脈不足
max=6.0:  F1=96.3%
max=7.0:  F1=96.8%  ← 最適
max=7.5:  F1=95.9%  ← ONNX 上限超過が散発
max=8.0:  F1=94.5%  ← ONNX エラー増加
max=9.0:  F1=87.7%  ← 大量のエラー
max=10.0: F1=82.3%  ← 壊滅的
```

`max_speech_duration` は VAD が「これ以上長いセグメントは作らない」という **ヒント** であり、厳密な上限ではありません。実際のセグメントは設定値より長くなることがあります。そのため Moonshine の 9 秒制限に対して `max=7` にマージンを持たせる必要があります。

### min_silence_duration

```
sil=0.70: F1=94.3%  ← 143セグメント（細切れ）
sil=1.00: F1=96.7%
sil=1.20: F1=96.8%  ← 最適
sil=1.50: F1=96.1%  ← max=7 で強制分割が増加
```

講義音声では句間の無音が 0.5〜1.0 秒程度なので、`sil=1.2` にすることで **文と文の間ではなく段落単位** でセグメントが作られます。

### threshold は鈍感

```
th=0.08〜0.17: F1=96.2〜96.8% でほぼ横ばい
```

Silero VAD の音声判定精度自体が高いため、閾値の影響は小さいです。

### 最終結果: TOP 5

| threshold | min_silence | min_speech | max_speech | F1 |
|-----------|-------------|------------|------------|------|
| **0.12** | **1.20** | **0.30** | **7.0** | **96.82%** |
| 0.15 | 1.20 | 0.30 | 7.0 | 96.80% |
| 0.13 | 1.00 | 0.28 | 7.0 | 96.74% |
| 0.15 | 1.30 | 0.30 | 7.0 | 96.63% |
| 0.12 | 1.10 | 0.30 | 7.0 | 96.60% |

11 分の講義音声に対して **F1 96.82%, Precision 98.94%, Recall 94.79%** を達成しました。リファレンスは Whisper Large-v3-turbo の SRT 出力です。

## 長時間音声への対応

72 分の大学講義で検証したところ、新たな課題が見つかりました。

### ハードカット + オーバーラップ + ファジーマージ

`max_speech_duration=7` でも、実際には 10〜15 秒のセグメントが生成されることがあります。そこで ASR に渡す前に **ハードカット**（8.5 秒上限）を適用します。

ただし単純に切ると単語の途中で切断されるため、**オーバーラップ付き分割 + テキストマージ** で対処します:

```
セグメント (12秒)
  → Chunk A: 0〜8.5秒
  → Chunk B: 7.6〜12秒  ← 0.9秒のオーバーラップ
  → 各チャンクを個別に文字起こし
  → テキストをマージ（重複部分を除去）
```

### 7 つのマージアルゴリズムを比較

マージアルゴリズム 7 種 × オーバーラップ幅 8 段階 = 全 56 組み合わせを検証しました:

| アルゴリズム | 方式 | ベスト F1 |
|------------|------|----------|
| concat | 単純結合（ベースライン） | 87.65% |
| suffix_prefix | 完全一致 suffix-prefix | 87.89% |
| lcs_substr | 最長共通部分文字列 | 87.59% |
| **fuzzy_sp** | **編集距離 25% 許容** | **87.91%** |
| ngram | 3-gram マッチング | 87.54% |
| half_cut | 中央でカット | 87.53% |
| best_of_3 | 上位3手法の最長結果 | 87.75% |

**fuzzy_suffix_prefix（編集距離 25% 許容）がすべてのオーバーラップ幅で首位** でした。ASR が同じ区間を微妙に異なる文字列で認識する（例:「データ」vs「テータ」）ケースを、ファジーマッチで吸収できるためです。

```python
def merge_fuzzy(texts, overlap_sec=0.9, threshold=0.15, search_mult=12):
    search = max(3, int(overlap_sec * search_mult))
    merged = texts[0]
    for t in texts[1:]:
        if not t:
            continue
        best_k = 0
        for k in range(2, min(len(merged), len(t), search) + 1):
            if edit_dist(merged[-k:], t[:k]) <= max(1, int(k * threshold)):
                best_k = k
        merged += t[best_k:]
    return merged
```

### 72 分講義の結果

| 条件 | セグメント数 | Precision | Recall | F1 | 処理時間 |
|------|-----------|-----------|--------|------|---------|
| overlap なし | 674 | 90.9% | 83.9% | 87.2% | 208秒 |
| **fuzzy_sp, 0.9秒 overlap** | 674 | 90.5% | 85.4% | **87.9%** | 220秒 |

CPU のみで 72 分の講義を **約 3.5 分で文字起こし**（RTF = 0.038, 約 26 倍速）。

## CJK スペース除去

Moonshine の日本語モデルは一部の出力で文字間にスペースを挿入します。以下の正規表現で CJK 文字間のスペースのみを除去します:

```python
import re

_CJK_SPACE_RE = re.compile(
    r'(?<=[\u3000-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef])'
    r' '
    r'(?=[\u3000-\u9fff\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef])'
)

def clean_text(text):
    return _CJK_SPACE_RE.sub('', text)
```

英語の単語間スペースには影響しません。

## Tiny JA vs Base JA

同一条件（最適 VAD パラメータ）での比較:

| モデル | サイズ | Precision | Recall | F1 | RTF (CPU) |
|--------|--------|-----------|--------|------|-----------|
| Moonshine Tiny JA | 69MB | 91.7% | 80.0% | 85.4% | 0.011 |
| **Moonshine Base JA** | **135MB** | **98.9%** | **94.8%** | **96.8%** | 0.025 |

Base JA が Tiny JA を **11 ポイント以上** 上回ります。135MB はスマートフォンでも十分実用的なサイズなので、特別な理由がなければ **Base JA を推奨** します。

## SRT 出力サンプル

最適パラメータで生成した字幕の一部:

```srt
1
00:00:01,940 --> 00:00:08,859
今回は宮崎県立看護大学の中尾博之先生にお話をいただきました

2
00:00:10,323 --> 00:00:17,468
高尾先生は数理統計学を専攻されましたが、その後、公衆衛生学や

3
00:00:17,652 --> 00:00:24,572
疫学の分野でデータの分析を中心に研究をされている先生です

4
00:00:25,684 --> 00:00:33,148
今回は、保健医療分野におけるデータサイエンスというテーマで、医療分野や
```

## ライセンスに関する注意

- **英語モデル**: MIT License（完全フリー）
- **日本語含む非英語モデル**: **Moonshine Community License**
  - 研究・非商用: 無制限 OK
  - 商用: 年商 100 万ドル（約 1.5 億円）未満なら OK（[要登録](https://moonshine.ai/community-license)）
  - 年商 100 万ドル以上: [エンタープライズライセンス](https://moonshine.ai/enterprise) が必要
  - 表示義務: **"Powered by Moonshine AI"** の掲示

## 最終推奨パラメータ一式

```python
import sherpa_onnx

# VAD
config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "silero_vad.onnx"
config.silero_vad.threshold = 0.12
config.silero_vad.min_silence_duration = 1.2
config.silero_vad.min_speech_duration = 0.3
config.silero_vad.max_speech_duration = 7.0
config.sample_rate = 16000
vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=300)

# ASR
recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
    encoder="moonshine-base-ja/encoder_model.ort",
    decoder="moonshine-base-ja/decoder_model_merged.ort",
    tokens="moonshine-base-ja/tokens.txt",
    num_threads=4,
)

# 安全ガード（長セグメント対策）
MAX_SAMPLES = int(16000 * 8.5)       # 8.5秒
MIN_SAMPLES = int(16000 * 0.15)      # 150ms
OVERLAP_SAMPLES = int(16000 * 0.90)  # 0.9秒
# マージ: fuzzy_suffix_prefix (編集距離 15%, search_mult=12)
```

## まとめ

| 項目 | 結果 |
|------|------|
| モデル | Moonshine Base JA (61.5M params, 135MB) |
| 精度 (11分講義) | F1 96.82% (vs Whisper Large-v3-turbo) |
| 精度 (72分講義) | F1 87.91% (fuzzy merge 込み) |
| 速度 | RTF 0.025〜0.038 (CPU, 26〜40 倍速) |
| 必須の知識 | 入力長上限 ~9秒、VoiceActivityDetector を使う |
| 最大の改善要因 | max_speech_duration=7 (10→7 で F1 +14.5%) |

Moonshine は Whisper の完全な代替ではありませんが、**エッジデバイスでのオフライン文字起こし** という用途では非常に強力な選択肢です。特に速度面での優位性は圧倒的で、CPU のみで 72 分の講義を 3.5 分で処理できます。

一方、Whisper Large-v3 と比較すると長時間音声での精度にはまだ差があり（F1 87.9% vs ほぼ 100%）、initial_prompt による文脈引き継ぎも非対応です。用途に応じた使い分けが重要です。

---

検証に使用したスクリプトは [GitHub](https://github.com/moonshine-ai/moonshine) の公式リポジトリと [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) の generate-subtitles.py を参考にしています。
