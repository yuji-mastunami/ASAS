from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.nn.modules.sparse import Embedding
import torch.optim as optim
from torch.optim import optimizer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import openpyxl
# functions

def exists(val):
    return val is not None

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers



# helper classes

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
        # self.hidden2tag = nn.Linear(hidden_dim2 + feature_dim + feature_dim + 2, tag_size)
        self.hidden2tag = nn.Linear(hidden_dim2 + feature_dim + feature_dim, tag_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.embed = nn.Embedding(num_tokens,dim)
        self.q_lineare = nn.Linear(dim,feature_dim)
        self.r_lineare = nn.Linear(dim,feature_dim)
        self.relu = nn.ReLU()

    # def forward(self, text, wc, sc, question, reference):
    def forward(self, text, question, reference):
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
        # question = self.sigmoid(question)
        question = self.relu(question)
        reference = self.r_lineare(reference)
        # reference = self.sigmoid(reference)
        reference = self.relu(reference)

        #  print("text:", text.size())
        #  print("question:", question.size())
        #  print("reference:", reference.size())
        #  print("wc:", wc.size())
        #  print("sc:", sc.size())
        # feature = torch.stack((wc,sc),1)
        # print("feature:", feature.size())

        # cat_list = [text, question, reference, feature]
        cat_list = [text, question, reference]
        hidden = torch.cat(cat_list, dim=1)
        ans = self.hidden2tag(hidden)
        ans = self.softmax(ans)

        return ans


def make_dict(dic, text_data):
    for t in text_data:
            t_list = t.split()
            for w in t_list:
                if w in dic: continue
                dic[w] = len(dic)
    return dic


def count(texts):
    max_len = 0
    for t in texts:
        t_lis = t.split()
        max_len = max(max_len, len(t_lis))
    return max_len

def make_index_list(l,d):
    max_len = 0
    return_list = []
    tmp_return_list = []
    for t in l:
        w_l = t.split()
        tmp_list = []
        for w in w_l:
            tmp_list.append(d[w])
        max_len = max(max_len, len(tmp_list))
        tmp_return_list.append(tmp_list)
    for tmp in tmp_return_list:
        for i in range(max_len- len(tmp)):
            tmp.insert(0,0)
        return_list.append(tmp)
    return return_list
    





def main():
    data = pd.read_table("./train.tsv/train.tsv")
    data_rel = pd.read_table("./train_rel_2.tsv/train_rel_2.tsv")
    question = pd.read_table("./question.tsv/question.tsv")
    reference = pd.read_table("./reference.tsv/reference.tsv")

    # データをつなげる
    data_list = [data, data_rel]
    data = pd.concat(data_list)

    # データから答案、問題、模範解答を取り出してリストに保存する
    texts_list = [t for t in data["EssayText"]]
    question_list = [ q for q in question["question"]]
    reference_list = [ r for r in reference["reference"]]

    word2index = {} # 単語辞書
    word2index.update({"<pad>": 0}) # padding用
    '''
    for t in texts_list:
        t_list = t.split()
        for w in t_list:
            if w in word2index: continue
            word2index[w] = len(word2index)
    '''
    for lis in [texts_list, question_list, reference_list]:
        word2index = make_dict(word2index, lis)

    text_len = count(texts_list)
    print(text_len)
    print(len(word2index))

    # パラメータ設定
    EMBEDDING_DIM = 768
    HIDDEN_DIM1 = 29462
    HIDDEN_DIM2 = 500
    DEPTH = 6
    SEQ_LEN = text_len
    TAG_SIZE = 4
    ATTN_DIM = 64
    VOCAB_SIZE = len(word2index)
    FEATURE_DIM = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    model = ASAS_Model(num_tokens= VOCAB_SIZE, dim=EMBEDDING_DIM, depth=DEPTH, seq_len=SEQ_LEN, attn_dim = ATTN_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2,feature_dim=FEATURE_DIM, tag_size=TAG_SIZE).to(device)

    model_path = "asasmodel.pth"
    model.load_state_dict(torch.load(model_path))

    sets = [s for s in data["EssaySet"]]
    scores = [s for s in data["Score1"]]

    question2index = make_index_list(question_list, word2index)
    reference2index = make_index_list(reference_list, word2index)

    # ワークブック作成
    wb = openpyxl.Workbook()
    sheet = wb.worksheets[0]
    file_name = "result.xlsx"

    sheet["A1"] = "text"
    sheet["B1"] = "score"
    sheet["C1"] = "predict"

    # wb.save(file_name)
    sheet_id = 2
    model.eval()
    with torch.no_grad():
        question_num = 1
        for i in range(len(texts_list)):
            if sets[i] == 1:
                print(texts_list[i])
                word_list = texts_list[i].split()
                input_index = []
                for w in word_list:
                    input_index.append(word2index[w])
                # 長さが足りない時はpadding
                if len(input_index) < text_len:
                    for _ in range(text_len- len(input_index)):
                        input_index.insert(0,0)
                # print("len:" ,len(input_index))
                input_tensor = torch.tensor([input_index], device=device)
                question_tensor = torch.tensor([question2index[0]], device=device)
                reference_tensor = torch.tensor([reference2index[0]], device=device)
                out = model(input_tensor, question_tensor, reference_tensor)
                _, predict = torch.max(out, 1)
                print("predict:", predict.item())
                print("score:", scores[i])

                sheet["A"+str(sheet_id)] = texts_list[i]
                sheet["B"+str(sheet_id)] = scores[i]
                sheet["C"+str(sheet_id)] = predict.item()
                sheet_id += 1
                
    wb.save(file_name)
                
            
    

if __name__ == "__main__":
    main()
