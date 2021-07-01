from random import randrange
from typing import ContextManager
import torch
import torch.nn.functional as F
from torch import nn, einsum
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import os
from torch.nn.modules.sparse import Embedding
import torch.optim as optim
from torch.optim import optimizer

def exists(val):
    return val is not None

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers
    
    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0.,1.) > prob_survival

    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False


    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, causal = False, act = nn.Identity(), init_eps = 1e-3):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x, gate_res = None):
        device, n = x.device, x.shape[1]

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            weight, bias = weight[:n, :n], bias[:n]
            mask = torch.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        attn_dim = None,
        causal = False,
        act = nn.Identity()
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None

        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        return x

# main classes

class gMLP(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        ff_mult = 4,
        attn_dim = None,
        prob_survival = 1.,
        causal = False,
        act = nn.Identity()
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity()

    def forward(self, x):
        x = self.to_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)

class ASAS_Model(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        ff_mult = 4,
        attn_dim = None,
        prob_servival = 1.,
        causal = False,
        act = nn.Identity(),
        hidden_dim1,
        hidden_dim2,
        feature_dim,
        tag_size
    ):
        super().__init__()
        self.gmlp = gMLP(num_tokens=num_tokens, dim=dim, depth=depth, seq_len=seq_len, ff_mult=ff_mult, attn_dim=attn_dim, prob_survival=prob_servival, causal=causal, act=act)
        self.hidden2hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden2tag = nn.Linear(hidden_dim2 + feature_dim + feature_dim + 2, tag_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.embed = nn.Embedding(num_tokens,dim)
        self.q_lineare = nn.Linear(dim,feature_dim)
        self.r_lineare = nn.Linear(dim,feature_dim)

    def forward(self, text, wc, sc, question, reference):
        text = self.gmlp(text)
        text = torch.mean(text, 1)
        # print(text.size())
        text = self.hidden2hidden(text)
        text = self.sigmoid(text)
        question = self.embed(question)
        question = torch.mean(question,1)
        reference = self.embed(reference)
        reference = torch.mean(reference,1)
        question = self.q_lineare(question)
        question = self.sigmoid(question)
        reference = self.r_lineare(reference)
        reference = self.sigmoid(reference)

        #  print("text:", text.size())
        #  print("question:", question.size())
        #  print("reference:", reference.size())
        #  print("wc:", wc.size())
        #  print("sc:", sc.size())
        feature = torch.stack((wc,sc),1)
        # print("feature:", feature.size())

        cat_list = [text, question, reference, feature]
        hidden = torch.cat(cat_list, dim=1)
        ans = self.hidden2tag(hidden)
        ans = self.softmax(ans)

        return ans






def train2batch(text, score, wc, sc, question, reference, batch_size=2):
    text_batch = [] 
    score_batch = [] 
    wc_batch = [] 
    sc_batch = []
    question_batch = []
    reference_batch = []

    text_shuffle, score_shuffle, wc_shuffle, sc_shuffle, question_shuffle, reference_shuffle = shuffle(text, score, wc, sc, question, reference)
    for i in range(0, len(text), batch_size):
        text_batch.append(text_shuffle[i:i+batch_size])
        score_batch.append(score_shuffle[i:i+batch_size])
        wc_batch.append(wc_shuffle[i:i+batch_size])
        sc_batch.append(sc_shuffle[i:i+batch_size])
        question_batch.append(question_shuffle[i:i+batch_size])
        reference_batch.append(reference_shuffle[i:i+batch_size])
    return text_batch, score_batch, wc_batch, sc_batch, question_batch, reference_batch



# データの読み込み
data = pd.read_table("./train.tsv/train.tsv")
data_rel = pd.read_table("./train_rel_2.tsv/train_rel_2.tsv")
question = pd.read_table("./question.tsv/question.tsv")
reference = pd.read_table("./reference.tsv/reference.tsv")

data = [data, data_rel]
data = pd.concat(data)
print(data)

texts = [t for t in data["EssayText"]]
word2index = {}
word2index.update({"<pad>":0})
for t in texts:
    text_list = []
    text_list = t.split()
    for w in text_list:
        if w in word2index: continue
        word2index[w] = len(word2index)

questions =[q for q in question["question"]]
references = [ r for r in reference["reference"]]

for text in questions:
    text_list = []
    text_list = text.split()
    for q in text_list:
        if q in word2index: continue
        word2index[q] = len(word2index)
for text in references:
    text_list = []
    text_list = text.split()
    for r in text_list:
        if r in word2index: continue
        word2index[r] = len(word2index)

print("vocab size:", len(word2index))

score = [s for s in data["Score1"]]
sets = [s for s in data["EssaySet"]]
question_list = [question["question"][s-1] for s in sets]

reference_list = [reference["reference"][s-1] for s in sets]

# textを単語idの配列にする
max_len = 0
index_texts_temp = []
word_cnt_list = []
sent_cnt_list = []
for text in texts:
    sentence2index = []
    text_list = []
    text_list = text.split()
    word_cnt_list.append(len(text_list))
    cnt = 0
    for w in text_list:
        sentence2index.append(word2index[w])
        for a in w:
            if '.' == a:
                cnt += 1
    max_len = max(max_len, len(sentence2index))
    index_texts_temp.append(sentence2index)
    sent_cnt_list.append(cnt)

index_texts = []
for index_list in index_texts_temp:
    for i in range(max_len - len(index_list)):     
        index_list.insert(0,0)
    index_texts.append(index_list)
text_len = max_len


# question,referenceに単語idをふる
index_tmp_question = []
index_tmp_reference = []
max_len = 0
for question in question_list:
    tmp_list = []
    tmp_list = question.split()
    index2question = []
    for w in tmp_list:
        index2question.append(word2index[w])
    index_tmp_question.append(index2question)
    max_len = max(max_len, len(index2question))
index_question = []
for index_list in index_tmp_question:
    for i in range(max_len - len(index_list)):
        index_list.insert(0,0)
    index_question.append(index_list)
max_len = 0
for ref in reference_list:
    tmp_list = []
    tem_list = ref.split()
    index2ref = []
    for w in tem_list:
        index2ref.append(word2index[w])
    index_tmp_reference.append(index2ref)
    max_len = max(max_len, len(index2ref))
index_reference = []
for index_list in index_tmp_reference:
    for i in range(max_len - len(index_list)):
        index_list.insert(0,0)
    index_reference.append(index_list)   
# シャフルするために一度dataframeに戻す
datasets = pd.DataFrame({'text': index_texts,'score':score,'word_cnt':word_cnt_list,'sent_cnt':sent_cnt_list,'question':index_question,'reference':index_reference})
datasets = datasets.sample(frac=1)

# 教師データとテストデータを分ける
text_data = [text for text in datasets["text"]]
score_data = [score for score in datasets["score"]]
wc_data = [ wc for wc in datasets["word_cnt"]]
sc_data = [sc for sc in datasets["sent_cnt"]]
question_data = [q for q in datasets["question"]]
reference_data = [r for r in datasets["reference"]]


train_text, test_text, train_score, test_score, train_wc, test_wc, train_sc, test_sc, train_question, test_question, train_reference, test_reference = train_test_split(text_data, score_data, wc_data, sc_data, question_data, reference_data, test_size=0.7)

# パラメータ設定
EMBEDDING_DIM = 768
HIDDEN_DIM1 = 29462
HIDDEN_DIM2 = 500
DEPTH = 36
SEQ_LEN = text_len
TAG_SIZE = 4
ATTN_DIM = 64
VOCAB_SIZE = len(word2index)
FEATURE_DIM = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
model = ASAS_Model(num_tokens= VOCAB_SIZE, dim=EMBEDDING_DIM, depth=DEPTH, seq_len=SEQ_LEN, attn_dim = ATTN_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2,feature_dim=FEATURE_DIM, tag_size=TAG_SIZE).to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses = []

for epoch in range(30):
    all_loss = 0
    text_batch, score_batch, wc_batch, sc_batch, question_batch, reference_batch = train2batch(train_text, train_score, train_wc, train_sc, train_question, train_reference)
    for i in range(len(text_batch)):
        batch_loss = 0
        model.train()

        # 配列を全てtensorにしてGPUにのせる
        text_tensor = torch.tensor(text_batch[i], device=device)
        score_tensor = torch.tensor(score_batch[i], device=device)
        wc_tensor = torch.tensor(wc_batch[i], device=device)
        sc_tensor = torch.tensor(sc_batch[i], device=device)
        question_tensor = torch.tensor(question_batch[i], device=device)
        reference_tensor = torch.tensor(reference_batch[i], device=device)

        # print(text_tensor.size())

        out = model(text_tensor, wc_tensor, sc_tensor, question_tensor, reference_tensor)
        batch_loss = loss_function(out, score_tensor)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        all_loss += batch_loss.item()
    print("epoch:", epoch+1, "loss", all_loss)

    model.eval()
    a = 0
    test_num = len(test_text)
    with torch.no_grad():
        text_batch, score_batch, wc_batch, sc_batch, question_batch, reference_batch = train2batch(test_text, test_score, test_wc, test_sc, test_question, test_reference)
        for i in range(len(text_batch)):
            text_tensor = torch.tensor(text_batch[i], device=device)
            score_tensor = torch.tensor(score_batch[i], device=device)
            wc_tensor = torch.tensor(wc_batch[i], device=device)
            sc_tensor = torch.tensor(sc_batch[i], device=device)
            question_tensor = torch.tensor(question_batch[i], device=device)
            reference_tensor = torch.tensor(reference_batch[i], device=device)

            out = model(text_tensor, wc_tensor, sc_tensor, question_tensor, reference_tensor)
            _,predicts = torch.max(out,1)
            for j,ans in enumerate(score_tensor):
                if predicts[j].item() == ans.item():
                    a += 1
                
                if i <= 10:
                    print("predicts:", predicts[j].item(), end='')
                    print("ans:", ans.item())
    print("acu:", a/test_num)   


    if all_loss < 0.1: break





        
print("done.")

'''
epoch: 1 loss 5320.203948445618
acu: 0.5564129301355579
done.
epoch: 2 loss 4778.660024050623
acu: 0.5794786235662148
done.
epoch: 3 loss 4718.57435497269
acu: 0.5777267987486966
done.
epoch: 4 loss 4705.92852845788
acu: 0.5684254431699687
done.
epoch: 5 loss 4692.73625879176
acu: 0.56
done.
epoch: 6 loss 4691.781143270433
acu: 0.5731803962460896
done.
epoch: 7 loss 4693.4400240033865
acu: 0.5734306569343066
done.
epoch: 8 loss 4690.1284789554775
acu: 0.5850677789363921
done.
epoch: 9 loss 4693.09296470508
acu: 0.574473409801877
done.
epoch: 10 loss 4692.969031978399
acu: 0.5827737226277372
done.
epoch: 11 loss 4684.2010492011905
acu: 0.586903023983316
done.
epoch: 12 loss 4686.96849668026
acu: 0.5846923879040667
done.
epoch: 13 loss 4683.998453602195
acu: 0.5667153284671533
done.
epoch: 14 loss 4678.930074958131
acu: 0.5883628779979145
done.
epoch: 15 loss 4675.918361160904
acu: 0.5883211678832116
done.
epoch: 16 loss 4681.630625437945
acu: 0.5851511991657977
done.
epoch: 17 loss 4685.615968424827
acu: 0.5597497393117831
done.
epoch: 18 loss 4687.597662612796
acu: 0.5857768508863399
done.
epoch: 19 loss 4679.832672148943
acu: 0.5834410844629823
done.
epoch: 20 loss 4683.140949949622
acu: 0.5850677789363921
done.
epoch: 21 loss 4684.165332678705
acu: 0.5610844629822732
done.
epoch: 22 loss 4680.625725373626
acu: 0.5864025026068822
done.
epoch: 23 loss 4685.0988990962505
acu: 0.5854848800834203
done.
epoch: 24 loss 4675.494038403034
acu: 0.5843587069864442
done.
epoch: 25 loss 4677.847975961864
acu: 0.5873618352450469
done.
epoch: 26 loss 4680.474071178585
acu: 0.5441501564129302
done.
epoch: 27 loss 4682.605218283832
acu: 0.5815224191866528
done.
epoch: 28 loss 4679.499918680638
acu: 0.5840667361835246
done.
epoch: 29 loss 4681.362691383809
acu: 0.5748905109489051
done.
epoch: 30 loss 4679.676956370473
acu: 0.5817309697601668
done.
'''