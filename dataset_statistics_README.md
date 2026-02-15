# Dataset Statistics Script

このスクリプトは、データセットの統計情報を出力します。

## 使用方法

### 全サブセットの分析

```bash
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config
```

これにより、データセットのメタデータから自動的に検出された全てのサブセットが分析されます。

### 特定のサブセットのみ分析

```bash
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config chat
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config writing
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config medical
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config science
python3 dataset_statistics.py NovelHacja/RubricHub_v1_config follow
```

### ヘルプ

```bash
python3 dataset_statistics.py --help
```

## 必要な依存関係

以下のパッケージが必要です：

```bash
pip install numpy datasets
```

または、プロジェクトの依存関係をインストール済みであれば、numpyのみを追加インストールしてください：

```bash
pip install numpy
```

## 出力内容

このスクリプトは、指定されたデータセットについて以下の統計情報を出力します：

### 各データセットごとの詳細統計

- **基本統計**
  - サンプル総数
  - 平均長 (characters)
  - 中央値
  - 分散
  - 標準偏差
  - 最小値/最大値

- **四分位情報**
  - Q1 (25パーセンタイル)
  - Q2 (50パーセンタイル/中央値)
  - Q3 (75パーセンタイル)
  - IQR (四分位範囲)

- **パーセンタイル分布**
  - 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%の各パーセンタイルでのプロンプト長

- **閾値分析**
  - 2,500文字、5,000文字、10,000文字、15,000文字、20,000文字、30,000文字を超えるサンプル数と割合

- **長さ分布（ビン）**
  - 0-1k, 1k-2.5k, 2.5k-5k, 5k-10k, 10k-20k, 20k-50k, 50k+ の各範囲に属するサンプル数と割合

### データセット間の比較表

複数のサブセットを分析した場合（第2引数を省略した場合）、全データセットを横断した比較表も出力されます：

- 各統計指標のデータセット間比較
- 閾値（2,500文字、10,000文字）を超えるサンプルの比較

単一のサブセットのみを分析した場合、比較表は表示されません。

## 実装の詳細

このスクリプトは、task.pyの`process()`メソッドで使用されているのと全く同じロジックでプロンプト長を計算します：

1. まず`extra_info`からJSON文字列を生成
2. それが失敗した場合は`prompt`、`reward_model`、`Rubrics`から生成
3. それも失敗した場合は`prompt`と`reward_model`のみから生成

これにより、実際の処理時と同じ方法でプロンプト長が計算されます。

### サブセットの自動検出

スクリプトは、Hugging Face Datasets APIの`get_dataset_config_names()`関数を使用して、データセットのメタデータから利用可能なサブセットを自動的に検出します。これにより、ハードコードされたリストを使用せず、どのようなデータセットにも対応できます。

## 注意事項

- このスクリプトは指定されたデータセットをダウンロードして処理するため、実行には時間がかかる場合があります
- インターネット接続が必要です（HuggingFace Hubからデータセットを読み込むため）
- 統計情報は標準出力に、ログメッセージは標準エラー出力に出力されます
