---
title: "Moonshine Voice ASR を sherpa-onnx で動かして日本語文字起こしする完全ガイド"
emoji: "🎙"
type: "tech"
topics: ["moonshine", "whisper", "asr", "sherpaonnx", "python"]
published: true
---

## はじめに

**Moonshine Voice** は、TensorFlow チームの初期メンバーが共同創業した Moonshine AI（旧 Useful Sensors）が開発するオープンソースの音声認識（ASR）モデルです。Hacker News で 316 ポイント（2026年2月時点）を獲得するなど注目を集めています。

英語では **Medium Streaming モデル (245M params) が Whisper Large v3 (1.55B params) と同等精度を約 1/6 のパラメータ数で実現** しています。さらにエッジデバイス（スマホ、Raspberry Pi）でリアルタイム動作することを前提に設計されています。

「じゃあ日本語はどうなの？」

この記事では、Moonshine の日本語モデル（Base JA, 61.5M params）を [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) で実際に動かし、VAD パラメータを含む **約 2,000 パターンのグリッドサーチ**（VAD 4 軸スイープ + マージアルゴリズム × オーバーラップ幅 × 編集距離閾値の全探索）で最適化した結果を共有します。

:::message alert
**本記事の精度指標について**

本記事に登場する F1 / Precision / Recall は、**Whisper Large-v3-turbo の認識結果**をリファレンスとした ASR 間比較です。人間による書き起こし（Ground Truth）ではありません。つまり「Moonshine が Whisper の出力にどれだけ一致するか」を測定しており、日本語認識の絶対精度ではない点にご注意ください。Moonshine Base JA の公式 CER（Character Error Rate: 文字誤り率、[FLEURS](https://huggingface.co/datasets/google/fleurs) データセット基準）は **13.62%** です。

また、テスト音声は非公開の講義音声のため再現実験はできません。結果は特定の音声条件下での参考値としてお読みください。
:::

## Moonshine のモデル世代を正しく理解する

Moonshine には **3 つの世代** があり、これを混同すると正確な議論ができません。

### 世代1: Original Moonshine（2024年10月）

- 論文: [arXiv:2410.15608](https://arxiv.org/abs/2410.15608)
- **英語のみ**、Tiny (27.1M params) / Base (61.5M params)
- Full-attention encoder + RoPE
- Whisper と違い **ゼロパディングなし** → 音声長に比例した計算量

### Flavors of Moonshine（2025年9月）

※本記事では便宜上「世代1.5」と呼びますが、公式名称ではありません。

- 論文: [arXiv:2509.02523](https://arxiv.org/abs/2509.02523)（**Tiny モデル**のみを対象）
- **日本語を含む 6 言語の単言語特化モデル**
- アーキテクチャは v1 と同じ（full-attention encoder）
- 同サイズの多言語モデルより大幅に高精度
- Base (61.5M) は論文発表の約7週間後（2025年10月）に HuggingFace で静かに公開。**論文での評価結果は存在しない**

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
| 他4言語 | Base | 61.5M | (省略) |

**日本語モデルは Flavors 世代（v1 アーキテクチャ）** です。v2 Streaming ではありません。

:::message alert
**sherpa-onnx の `from_moonshine_v2()` は紛らわしい命名です。** これは Moonshine v2（Streaming）とは無関係で、ONNX エクスポート形式の世代（4ファイル → 2ファイル）を指します。英語の Tiny/Base も同じ `from_moonshine_v2()` で使えることがその証拠です。経緯は [sherpa-onnx Issue #3223](https://github.com/k2-fsa/sherpa-onnx/issues/3223) → [PR #3232](https://github.com/k2-fsa/sherpa-onnx/pull/3232) を参照。
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
# クローン後、パスを短くリネーム（以降のコードはこの名前を前提）
mv sherpa-onnx-moonshine-base-ja-quantized-2026-02-27 moonshine-base-ja

# Tiny JA (69MB, 高速)
git clone https://huggingface.co/csukuangfj2/sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27
mv sherpa-onnx-moonshine-tiny-ja-quantized-2026-02-27 moonshine-tiny-ja
```

Silero VAD モデルも必要です:

```bash
# sherpa-onnx のリリースページから取得（推奨）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
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

sherpa-onnx の ONNX 量子化モデルで Moonshine を使用する場合、**入力長が約 10 秒を超えると ONNX ランタイムがエラーを返す** ことがあります。これは Moonshine 自体の制約ではなく（公式 README は「30 秒以下推奨」、原論文では 10〜55 秒のクリップでテスト済み）、**ONNX エクスポート + 量子化に起因する制約** と考えられます。筆者の実験で確認した経験的な値です。エラーメッセージ:

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
    config.silero_vad.max_speech_duration = 7.5
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

※F1 は Whisper 出力との一致度（前述の通り ASR 間比較）。

**必ず `VoiceActivityDetector` を使ってください。**

## VAD パラメータチューニング

### Moonshine の入力長制約

| サンプル数 | 秒数 | 結果 |
|-----------|------|------|
| 128,000 | 8.0s | OK |
| 144,000 | 9.0s | OK（ギリギリ） |
| 158,464 | 9.9s | **ONNX エラー** |

9.0 秒はギリギリ通りますが、余裕がありません。最短側にも制約があり、約 0.1 秒未満ではエラーになります。安全な範囲は **0.15 秒〜8.5 秒** です。

### 4 つのパラメータ

| パラメータ | 説明 | デフォルト | 最適値 |
|-----------|------|---------|--------|
| `threshold` | 音声/非音声の判定閾値 | 0.5 | **0.12** |
| `min_silence_duration` | この秒数の無音でセグメント確定 | 0.25 | **1.2** |
| `min_speech_duration` | この秒数未満の発話を無視 | 0.25 | **0.3** |
| `max_speech_duration` | セグメントの最大長 | 5.0 | **7.5** |

### max_speech_duration が最も重要

4 軸の中で **`max_speech_duration` が精度に最も大きな影響** を与えます。

初期検証（11分音声、ハードカットなし）での F1 推移:

```
max=5.0:  F1=95.3%  ← セグメントが短すぎて文脈不足
max=6.0:  F1=96.3%
max=7.0:  F1=96.8%  ← ハードカットなしでは最適
max=7.5:  F1=95.9%  ← ONNX 上限超過が散発
max=8.0:  F1=94.5%  ← ONNX エラー増加
max=9.0:  F1=87.7%  ← 大量のエラー
max=10.0: F1=82.3%  ← 壊滅的
```

`max_speech_duration` は VAD が「これ以上長いセグメントは作らない」という **ヒント** であり、厳密な上限ではありません。実際のセグメントは設定値より長くなることがあります。

:::message
**ハードカット + オーバーラップ + ファジーマージを導入した後の再最適化では `max=7.5` が最良**になりました。8.5秒ハードカットが安全網になるため、VAD 側のマージンを緩和できます。本記事の最終推奨値は **7.5** です。
:::

### min_silence_duration

```
sil=0.70: F1=94.3%  ← 143セグメント（細切れ）
sil=1.00: F1=96.7%
sil=1.20: F1=96.8%  ← 最適
sil=1.50: F1=96.1%  ← 強制分割が増加
```

講義音声では句間の無音が 0.5〜1.0 秒程度なので、`sil=1.2` にすることで **文と文の間ではなく段落単位** でセグメントが作られます。

### threshold は鈍感

```
th=0.08〜0.17: F1=96.2〜96.8% でほぼ横ばい
```

Silero VAD の音声判定精度自体が高いため、閾値の影響は小さいです。

### VAD 4 軸スイープ結果: TOP 5（ハードカットなし時点）

以下は初期検証（11 分音声、ハードカット未導入）でのVAD パラメータ TOP 5 です:

| threshold | min_silence | min_speech | max_speech | F1 |
|-----------|-------------|------------|------------|------|
| **0.12** | **1.20** | **0.30** | **7.0** | **96.82%** |
| 0.15 | 1.20 | 0.30 | 7.0 | 96.80% |
| 0.13 | 1.00 | 0.28 | 7.0 | 96.74% |
| 0.15 | 1.30 | 0.30 | 7.0 | 96.63% |
| 0.12 | 1.10 | 0.30 | 7.0 | 96.60% |

11 分の講義音声に対して **F1 96.82%, Precision 98.94%, Recall 94.79%**（Whisper Large-v3-turbo 出力との LCS ベース文字単位比較）。

この後、ハードカット + オーバーラップ + ファジーマージを導入した上で再最適化を行い、`max_speech_duration` は **7.0 → 7.5** に更新されました（8.5秒ハードカットが安全網になるため）。

## 長時間音声への対応

72 分の大学講義で検証したところ、新たな課題が見つかりました。

### ハードカット + オーバーラップ + ファジーマージ

`max_speech_duration=7.5` でも、実際には 10〜15 秒のセグメントが生成されることがあります。そこで ASR に渡す前に **ハードカット**（8.5 秒上限）を適用します。

ただし単純に切ると単語の途中で切断されるため、**オーバーラップ付き分割 + テキストマージ** で対処します:

```
セグメント (12秒)
  → Chunk A: 0〜8.5秒
  → Chunk B: 7.6〜12秒  ← 0.9秒のオーバーラップ
  → 各チャンクを個別に文字起こし
  → テキストをマージ（重複部分を除去）
```

### 7 つのマージアルゴリズムを比較

初期検証として、マージアルゴリズム 7 種 × オーバーラップ幅 8 段階 = 全 56 組み合わせを検証しました（72 分講義音声）:

| アルゴリズム | 方式 | ベスト F1 |
|------------|------|----------|
| concat | 単純結合（ベースライン） | 87.65% |
| suffix_prefix | 完全一致 suffix-prefix | 87.89% |
| lcs_substr | 最長共通部分文字列 | 87.59% |
| **fuzzy_sp** | **編集距離 25% 許容** | **87.91%** |
| ngram | 3-gram マッチング | 87.54% |
| half_cut | 中央でカット | 87.53% |
| best_of_3 | 上位3手法の最長結果 | 87.75% |

**fuzzy_suffix_prefix が全オーバーラップ幅で首位** でした。ASR が同じ区間を微妙に異なる文字列で認識する（例:「データ」vs「テータ」）ケースを、ファジーマッチで吸収できるためです。

上記テーブルの fuzzy_sp は初期検証で編集距離閾値を 25% に固定して比較したものです。その後、全パラメータ一括グリッドサーチ（overlap 12段階 × max_dur 5段階 × マージ設定 34通り = 約 2,000 パターン、閾値は 15%〜45% の 7 段階で探索）を実施し、最終的に **overlap=0.90秒、編集距離閾値 15%** が最適と確定しました。グリッドサーチは Optuna 等の実験管理ツールではなく、自作の Python スクリプト（`ThreadPoolExecutor` で 16 スレッド並列実行、4-gram F1 で高速スクリーニング → TOP 20 を正式 LCS F1 で再検証する 2 段階方式）で実施しました。

```python
def edit_dist(a, b):
    """レーベンシュタイン距離（DP）"""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def merge_fuzzy(texts, overlap_sec=0.90, threshold=0.15, search_mult=12):
    """ファジー suffix-prefix マージ"""
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
| **fuzzy_sp, 0.90秒 overlap** | 674 | 90.5% | 85.4% | **87.9%** | 220秒 |

CPU のみで 72 分の講義を **約 3.5 分で文字起こし**（RTF = 0.038, 約 26 倍速）。

:::message
72分と11分で F1 に大きな差（87.9% vs 96.8%）がある理由は、音声の違いによるものです。72分音声は話者交代、質疑応答、無音区間が多く、11分音声よりも複雑な構成です。また、リファレンス自体が Whisper 出力のため、Whisper の誤認識もスコアに影響します。
:::

### テスト環境と並列処理

上記の処理時間は以下の環境で測定しています:

| 項目 | スペック |
|------|---------|
| PC | ASUS Zenbook 14 (UX3405CA) |
| CPU | Intel Core Ultra 9 285H (16コア / 16スレッド) |
| RAM | 32GB |
| GPU | **不使用**（CPU 推論のみ） |
| OS | Windows 11 |

処理速度の大きな要因として、**16 スレッド並列 ASR** があります。セグメントごとに独立した Recognizer インスタンスを `queue.Queue` で排他管理し、`ThreadPoolExecutor` で同時処理しています:

```python
import queue
from concurrent.futures import ThreadPoolExecutor

class ASRPool:
    def __init__(self, n_workers=16):
        self._rec_queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=n_workers)
        # 各ワーカーに専用の Recognizer を割り当て（num_threads=1）
        for _ in range(n_workers):
            self._rec_queue.put(
                sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                    encoder="moonshine-base-ja/encoder_model.ort",
                    decoder="moonshine-base-ja/decoder_model_merged.ort",
                    tokens="moonshine-base-ja/tokens.txt",
                    num_threads=1,  # Recognizer 内部は 1 スレッド
                )
            )

    def recognize_one(self, samples):
        rec = self._rec_queue.get()      # 空きが出るまでブロック
        try:
            stream = rec.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            rec.decode_stream(stream)
            return clean_text(stream.result.text.strip())
        finally:
            self._rec_queue.put(rec)      # 使い終わったら返却
```

ポイント:
- Recognizer ごとに `num_threads=1` とし、**並列度はプロセスレベルで制御**
- sherpa-onnx の推論は内部で C++ ネイティブコードが走るため、Python の GIL の影響を受けない
- 全セグメントを一括で `pool.submit()` に投入し、`as_completed()` で結果を回収

1 スレッド（逐次処理）で同じ 72 分音声を処理すると、16 スレッド並列に比べて約 4〜5 倍の時間がかかります。コア数の少ない環境では RTF がこの数値より大きくなる点にご注意ください。

## CJK スペース除去

Moonshine の日本語モデルは一部の出力で CJK 文字間にスペースを挿入します。これはモデルの学習データに由来する挙動と考えられます（トークナイザがサブワード境界にスペースを挿入するケースなど）。以下の正規表現で CJK 文字間のスペースのみを除去します:

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

※F1 は Whisper 出力との一致度（ASR 間比較）。

Base JA が Tiny JA を **11 ポイント以上** 上回ります。公式 CER（FLEURS）でも Tiny JA は 17.87% に対し Base JA は 13.62% と差があります。135MB はスマートフォンでも十分実用的なサイズなので、特別な理由がなければ **Base JA を推奨** します。

## 出力の特徴と注意点

Moonshine Base JA の出力にはいくつかの傾向があります:

- **固有名詞の認識精度が低い**: 人名や地名が誤認識されることが多い（例: 「中尾」→「高尾」）。Whisper の `initial_prompt` のような文脈ヒント機能は Moonshine にはない
- **CJK 文字間スペース**: 前述の正規表現で除去が必要
- **句読点なし**: Moonshine の日本語モデルは句読点を出力しない。必要なら後処理で追加する必要がある
- **長文の文脈理解**: セグメント内の文脈は保持されるが、セグメント間の文脈引き継ぎはない

## ライセンスに関する注意

- **英語モデル**: MIT License（完全フリー）
- **日本語含む非英語モデル**: **Moonshine Community License**
  - 研究・非商用: 無制限 OK
  - 商用: 年商 100 万ドル（約 1.5 億円）未満なら OK（[要登録](https://moonshine.ai/community-license)）
  - 年商 100 万ドル以上: [エンタープライズライセンス](https://moonshine.ai/enterprise) が必要
  - 詳細はライセンス全文を確認してください

## 最終推奨パラメータ一式

```python
import sherpa_onnx

# VAD
config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "silero_vad.onnx"
config.silero_vad.threshold = 0.12
config.silero_vad.min_silence_duration = 1.2
config.silero_vad.min_speech_duration = 0.3
config.silero_vad.max_speech_duration = 7.5   # ハードカット併用前提
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
MAX_SAMPLES = int(16000 * 8.5)       # 8.5秒ハードカット
MIN_SAMPLES = int(16000 * 0.15)      # 150ms 未満はスキップ
OVERLAP_SAMPLES = int(16000 * 0.90)  # 0.9秒オーバーラップ
FUZZY_TH = 0.15                      # 編集距離閾値 15%
SEARCH_MULT = 12                     # 検索幅倍率
```

## まとめ

| 項目 | 結果 |
|------|------|
| モデル | Moonshine Base JA (Flavors 世代, 61.5M params, 135MB) |
| 公式 CER | 13.62% (FLEURS, 人間リファレンス) |
| Whisper 出力との一致度 (11分) | F1 96.82% (ASR 間比較) |
| Whisper 出力との一致度 (72分) | F1 87.91% (fuzzy merge 込み) |
| 速度 | RTF 0.025〜0.038 (CPU 16スレッド並列, 26〜40 倍速) |
| テスト環境 | ASUS Zenbook 14 / Core Ultra 9 285H / 32GB RAM |
| 必須の知識 | ONNX量子化モデルの入力長上限 ~10秒、VoiceActivityDetector を使う |
| 最大の改善要因 | max_speech_duration 調整 (10→7.5 で大幅改善) |

Moonshine は Whisper の完全な代替ではありませんが、**エッジデバイスでのオフライン文字起こし** という用途では非常に強力な選択肢です。特に速度面での優位性は圧倒的で、CPU のみで 72 分の講義を 3.5 分で処理できます。

一方で以下の制約があります:

- **日本語精度は英語ほどではない**: 公式 CER 13.62% は Whisper Large-v3 の日本語精度には及ばない
- **固有名詞に弱い**: `initial_prompt` のような文脈ヒント機能がない
- **Flavors 論文は Tiny のみ**: Base JA は論文での評価がなく、公式ベンチマークは FLEURS の CER のみ

用途に応じた使い分けが重要です。速度重視・オフライン前提なら Moonshine、精度重視なら Whisper という棲み分けになるでしょう。

---

検証に使用したスクリプトは [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) の generate-subtitles.py を参考にしています。Moonshine の公式リポジトリは [GitHub](https://github.com/moonshine-ai/moonshine) にあります。
