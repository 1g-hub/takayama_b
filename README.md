# takayama

# 研究テーマ (B3 ~ )

「自然言語処理と深層学習に基づいた 4 コマ漫画のセリフの感情推定」

## B3 実験

- [資料](https://github.com/1g-hub/takayama/tree/master/JK_EX_2)

- Embedding

UMARU にあげられていた大規模コーパス (wiki_and_narou_and_aozora_surface_u_jumanpp.txt) を用いて学習させた Doc2Vec モデル (d2v_wiki.model) を使用してセリフの文のベクトルを求めた.

| pamameters | value |
| :--: | :--: |
|vec_size|300|
|epochs|30|
|window_size|8|
|min_count|5|
|alpha|0.05|
|workers|4|

- Net

LSTM => MLP

パラメータは [資料](https://github.com/1g-hub/takayama/tree/master/JK_EX_2) 参照のこと.

- LSTM への入力

各セリフ 1 文について, そのセリフを末尾とする同一 4 コマ内に属し, 連続した n 個のセリフの文ベクトル　[batch x n x 300]

参照する前のセリフがない場合はその分散表現を零ベクトルとする "\<pad\>" を置くことで対処した.

- MLP からの出力

末尾のセリフに対応する感情ラベル (0: 喜楽, 1: その他) [batch x 2]

softmax をかけて得られた結果が出力される.

## B4 (前期)
