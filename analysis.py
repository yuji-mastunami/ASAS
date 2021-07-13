import openpyxl
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np


def calc_acc(ws):
    cnt = 0
    ok = 0

    for fact in ws:
        if fact[1].value == fact[2].value:
            ok += 1
        cnt += 1
    return ok / cnt

def count_word(ws):
    correct_list = []
    mistake_list = []

    for fact in ws:
        if fact[1].value == fact[2].value:
            correct_list.append(len(fact[0].value))
        else:
            mistake_list.append(len(fact[0].value))
    correct_len = sum(correct_list) / len(correct_list)
    mistake_len = sum(mistake_list) / len(mistake_list)

    return correct_len, mistake_len

def count_sentence(ws):
    correct_list = []
    mistake_list = []    
    for fact in ws:
        sentence_list = fact[0].value.split()
        if fact[1].value == fact[2].value:
            correct_list.append(len(sentence_list))
        else:
            mistake_list.append(len(sentence_list))
    correct_len = sum(correct_list) / len(correct_list)
    mistake_len = sum(mistake_list) / len(mistake_list)  
    return correct_len, mistake_len

def make_confusion_matrix(ws):
    score_list = []
    predict_list = []
    cnt = 0
    for fact in ws:
        if cnt != 0:
            score_list.append(int(fact[1].value))
            predict_list.append(int(fact[2].value))
        cnt += 1
    score_np = np.array(score_list)
    predct_np = np.array(predict_list)
    return confusion_matrix(score_np, predct_np)

def calc_kappa_score(ws):
    score_list = []
    predict_list = []
    cnt = 0
    for fact in ws:
        if cnt != 0:
            score_list.append(int(fact[1].value))
            predict_list.append(int(fact[2].value))
        cnt += 1
        
    score_np = np.array(score_list)
    predict_np = np.array(predict_list)
    return cohen_kappa_score(score_np, predict_np, weights="quadratic")

    
    

def main():
    file_name = "result_"
    # print("question_num : ",end="")
    # question_num = input()
    # エクセル読み込み
    for question_num in range(1,11):
        wb = openpyxl.load_workbook(file_name+str(question_num)+".xlsx")
        # シート選択
        ws = wb["Sheet"]

        # fact = ws[2]
        # print(type(fact[0].value))

        # print(fact[0].value)
        '''
        for f in fact:
            print(f.value)
        
        acc = calc_acc(ws)
        print("question:", question_num,"acc:", acc)
        
        correct_len, mistake_len = count_word(ws)
        print("qestion:", question_num, "correct:", correct_len, "mistake:", mistake_len)
        '''
        # correct_len, mistake_len = count_sentence(ws)
        # print("qestion:", question_num, "correct:", correct_len, "mistake:", mistake_len)
        # cm = make_confusion_matrix(ws)
        print("question:", question_num, "QWK:", calc_kappa_score(ws))
        
        
        

if __name__ == "__main__":
    main()