import os
import codecs
import pandas as pd 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.optim as optim
import torch.nn.functional as F

# GPUを使うためのやつ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lstm(bach)版
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, feature_dim,linear_size):
        # 親クラスのコンストラクタ
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # <pad>の単語IDが0なので,padding_id=0としている
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # batch_first=Trueでバッチサイズx文章の長さxベクトルの次元数になる
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, linear_size)
        self.dropout = nn.Dropout(p=0.2)
        self.hidden2tag = nn.Linear(linear_size + feature_dim, target_size)
        self.softmax = nn.LogSoftmax()

    # 順伝播処理
    def forward(self, sentence, feature):
        # 文章内の各単語をベクトルに変換して出力する
        embeds = self.word_embeddings(sentence)
        _, lstm_out = self.lstm(embeds)
        out1 = self.linear(lstm_out[0])
        out1 = torch.sigmoid(out1)
        # print("out",out1.size())
        # print("feature",feature.view(1,100,3).size())
        # out torch.Size([1, 100, 150])
        # feature torch.Size([1, 100, 3])
        batch_size = feature.size()[0]
        merge = torch.cat([out1,feature.view(1,batch_size,3)],axis=2)
        merge = self.dropout(merge)
        # print("merge",merge.size())\\
        # merge torch.Size([1, 100, 1
        tag_space = self.hidden2tag(merge)
        tag_scores = self.softmax(tag_space.squeeze())

        return tag_scores



# lstmの定義
# class LSTMClassifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, feature_dim):
#         # 親クラスのコンストラクタ
#         super(LSTMClassifier, self).__init__()
#         # 隠れ層の次元数
#         self.hidden_dim = hidden_dim
#         # 入力単語をベクトルに変換
#         self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
#         # lstmの隠れ層
#         self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
#         self.linear = nn.Linear(hidden_dim, 150)
#         # feature_dimとhidden_dimをconcatしたものをsoftmaxに食わせるための線形結合
#         self.hidden2tag = nn.Linear(150 + feature_dim, target_size)
#         self.softmax = nn.LogSoftmax(dim=1)
        
#     # 順伝播処理
#     def forward(self, sentence, feature):
#         # 文章内の各単語をベクトルに変換して出力する
#         embeds = self.word_embedding(sentence)
#         # 2次元テンソルを３次元テンソルに変えてLSTM に流す
#         _, lstm_out = self.lstm1(embeds.view(len(sentence), 1 ,-1))
#         # lstm_out[0]は３次元テンソルになっているので２次元に戻す
#         out1 = self.linear(lstm_out[0].view(-1, self.hidden_dim))
#         # merge = torch.cat([lstm_out[0].view(-1, self.hidden_dim),feature], axis=1)
#         out1 = torch.sigmoid(out1)
#         # print("out1", out1.size())
        
#         # print("feature", feature.view(1,3).size())
#         merge = torch.cat([out1,feature.view(1,3)], axis=1)
#         tag_space = self.hidden2tag(merge)
#         tag_scores = self.softmax(tag_space)
#         return tag_scores




# read .tsv
data = pd.read_table("train.tsv/train.tsv")
data1 = pd.read_table("train_rel_2.tsv/train_rel_2.tsv")


data = [data, data1]
data = pd.concat(data)
print(data)
print(type(data))
essay_text = data["EssayText"]
essay_score = data["Score1"]
essay_set = data["EssaySet"] 

# 解答のリスト
texts = [t for t in essay_text]
# 点数のリスト
labels = [l for l in essay_score]

word2index = {}
# 系列をそろえるための文字列を追加
word2index.update({"<pad>":0})
# 単語ID辞書を作成する
for sentence in texts:
    sentence_list = []
    sentence_list = sentence.split()
    for s in sentence_list:
        if s in word2index: continue
        word2index[s] = len(word2index)
print("vocab size: ", len(word2index))

# vocab size:  29403

# 文章を単語IDの列に変換したい
def sentence2index(sentence):
    sentence_list = []
    sentence_list = sentence.split()
    return [word2index[s] for s in sentence_list]


# ピリオドの数を調べる
def count_period(sentence):
    c = 0
    for s in sentence:
        if s == '.':
            c += 1
    return c
    
    
# 問題番号のリスト
sets = [s for s in essay_set]
# 解答の単語数のリスト
word_count = [len(sentence2index(s)) for s in essay_text]
# 解答の文の数のリスト
sent_count = [count_period(s) for s in essay_text]

# print(texts[0])
# print(sent_count[0])


datasets = pd.DataFrame(columns=["text","score", "sets", "word_count", "sent_count"])
for i in range(len(texts)):
    s = pd.Series([texts[i], labels[i], sets[i], word_count[i], sent_count[i]], index=datasets.columns)
    datasets = datasets.append(s, ignore_index=True)

# datasets = datasets.sample(frac=1).reset_index(drop=True)
# datasets.head()
print(datasets)

index_dataset_text_tmp = []
index_dataset_score = []
index_dataset_features = []
# 系列の長さの最大値を取得
max_len = 0
for text, score, sets, word_count, sent_count in zip(datasets["text"],datasets["score"],datasets["sets"],datasets["word_count"],datasets["sent_count"]):
    index_text = sentence2index(text)
    index_score = [score]
    index_features = []
    index_features.append(sets)
    index_features.append(word_count)
    index_features.append(sent_count)
    # index_features = torch.stack([sets, word_count, sent_count], dim=0)
    index_dataset_text_tmp.append(index_text)
    index_dataset_score.append(index_score)
    index_dataset_features.append(index_features)
    if max_len < len(index_text):
        max_len = len(index_text)

# print(max_len)
# 402

# 系列をそろえるためのパディング追加
index_dataset_text = []
for text in index_dataset_text_tmp:
    for i in range(max_len - len(text)):
        text.insert(0,0)
    index_dataset_text.append(text)

train_text, test_text, train_score, test_score, train_features, test_features = train_test_split(index_dataset_text, index_dataset_score, index_dataset_features, train_size=0.7)


# データをバッチでまとめる
def train2batch(text, score, features, batch_size=64):
    text_batch = []
    score_batch = []
    features_batch = []
    text_shuffle, score_shuffle, features_shuffle = shuffle(text, score, features)
    for i in range(0, len(text), batch_size):
        text_batch.append(text_shuffle[i:i+batch_size])
        score_batch.append(score_shuffle[i:i+batch_size])
        features_batch.append(features_shuffle[i:i+batch_size])
    return text_batch, score_batch, features_batch








def score2tensor(score):
    return torch.tensor([score], dtype=torch.long)


# # 確認用
# print(sentence2index("Some additional information that we would need to replicate the experiment is how much vinegar should be placed in each identical container, how or what tool to use to measure the mass of the four different samples and how much distilled water to use to rinse the four samples after taking them out of the vinegar."))

# 全単語数の取得
VOCAB_SIZE = len(word2index)
# 単語ベクトルの次元数
EMBEDDING_DIM = 200
# 隠れ層の次元数
HIDDEN_DIM = 128
# 点数の数
SCORE_SIZE = 4
FEATURE_DIM = 3
LINEARE_SIZE = 150

# to(device)でGPU対応させる
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, SCORE_SIZE, FEATURE_DIM, LINEARE_SIZE).to(device)
# # データを7:3に分ける
# traindata, testdata = train_test_split(datasets, train_size=0.7)

# 損失関数
loss_function = nn.NLLLoss()
# 最適化
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
for epoch in range(100):
    all_loss = 0
    text_batch, score_batch, features_batch = train2batch(train_text, train_score, train_features)
    acu = 0
    for i in range(len(text_batch)):
        bach_loss = 0

        model.zero_grad()

        # 順伝播させるtensorはGPUに処理させるのでGPUにセットする
        text_tensor = torch.tensor(text_batch[i], device=device)
        score_tensor = torch.tensor(score_batch[i], device=device)
        features_tesor = torch.tensor(features_batch[i], device=device)

        out = model(text_tensor,features_tesor)

        batch_loss = loss_function(out, score_tensor.squeeze())
        batch_loss.backward()
        optimizer.step()

        all_loss += batch_loss.item()

        
        _,predict = torch.max(out,1)
        for j, ans in enumerate(score_tensor):
            if predict[j].item() == ans.item():
                acu += 1
    print("epoch", epoch, "\t", "loss", all_loss, "\t","accuracy",acu/len(train_text))
    if all_loss < 0.1: break
print("done.")

test_num = len(test_text)
a = 0
with torch.no_grad():
    text_batch, score_batch, features_batch = train2batch(test_text, test_score, test_features)
    for i in range(len(text_batch)):
        text_tensor = torch.tensor(text_batch[i], device=device)
        score_tensor = torch.tensor(score_batch[i], device=device)
        features_tesor = torch.tensor(features_batch[i],device=device)

        out = model(text_tensor,features_tesor)
        _, predicts = torch.max(out, 1)
        for j ,ans in enumerate(score_tensor):
            if predicts[j].item() == ans.item():
                a += 1
print("predict:", a/test_num)
# # 各エポックの合計のloss値を格納する
# losses = []
# for epoch in range(10):
#     all_loss = 0
#     acu = 0
#     for text, score, sets, word_count, sent_count in zip(traindata["text"],traindata["score"],traindata["sets"],traindata["word_count"],traindata["sent_count"]):
#         # モデルが持っている勾配の情報をリセット
#         model.zero_grad()
#         # 文章を単語IDの系列に変換
#         inputs = sentence2index(text)
#         # 問題番号、単語数、文の数をテンソルにしてまとめる
#         sets = score2tensor(sets)
#         word_count = score2tensor(word_count)
#         sent_count = score2tensor(sent_count)
#         features = torch.stack([sets, word_count, sent_count], dim=0)
        
#         # 順伝播の結果を受け取る
#         out = model(inputs, features)
#         # 正解をテンソルに変換
#         answer = score2tensor(score)
#         # print(answer)
#         # lossを計算
#         loss = loss_function(out, answer)
#         # 勾配をセット
#         loss.backward()
#         # 逆伝播でパラメータ更新
#         optimizer.step()
#         # lossを集計
#         all_loss += loss.item()
#         _, predict = torch.max(out, 1)
#         if predict == answer:
#             acu += 1
        
#     losses.append(all_loss/len(traindata))
#     print("epoch:", epoch,"\t", "loss:", all_loss,"\t", "accuracy:",acu/len(traindata))
# print(losses)

# epoch 0          loss 257.1193224787712          accuracy 0.5279666319082378
# epoch 1          loss 183.73179191350937         accuracy 0.6765797705943691
# epoch 2          loss 139.09129652380943         accuracy 0.7687174139728884
# epoch 3          loss 99.75755408406258          accuracy 0.848800834202294
# epoch 4          loss 71.01936015486717          accuracy 0.8962669447340981
# epoch 5          loss 47.923580415546894         accuracy 0.9343899895724713
# epoch 6          loss 34.92510147020221          accuracy 0.9533680917622523
# epoch 7          loss 29.33976041339338          accuracy 0.9592075078206465
# epoch 8          loss 21.680286914110184         accuracy 0.970385818561001
# epoch 9          loss 16.8548723468557   accuracy 0.9751824817518249
# done.
# predict: 0.8548905109489051