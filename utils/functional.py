import random

import torch
import numpy as np
import math

import copy


def select_past_key_value(past_key_value):
    present = ()
    for layer_past in zip(*past_key_value):
        key, value = tuple(zip(*layer_past))
        key = torch.cat(key, dim=2)
        value = torch.cat(value, dim=2)
        present += ((key, value), )

    return present

def expand_past_key_value(past_key_value, class_num):
    """
    Input sentence's batch is 1. To use the past_key_value for multiple answers, we need to expand the key and value's batch to the class number.
    """
    present = ()
    for layer_past in past_key_value:
        key = layer_past[0]
        value = layer_past[1]
        present += ((key.expand(class_num, -1, -1, -1), value.expand(class_num, -1, -1, -1)), )

    return present

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def select_order_by_distribution(examples,values):
    assert len(examples) == len(values)
    num_examples = len(examples)
    buckets = [ [] for _ in range(values[0].shape[0])]
    for i,value in enumerate(values):
        buc = np.argmax(value)
        buckets[buc].append(i)
    for i in range(len(buckets)):
        buckets[i].sort(key=lambda x: values[x][i],reverse=True)
        
    # print(buckets)
    # for x in buckets:
    #     print([values[i] for i in x])
    ans = sum(values)
    order = []
    # print(ans)
    for _ in range(num_examples):
        bucs = ans.argsort()
        p = len(bucs)-1
        while len(buckets[ bucs[p] ])==0:
            p-=1
        order.append(buckets[ bucs[p] ][0])
        buckets[ bucs[p] ].pop(0)
        ans-=values[ order[-1] ]
        # print(ans)
        
    result=[]
    for i in order:
        result.append(examples[i])
    return result

def select_order_by_bucket(examples,values):
    assert len(examples) == len(values)
    num_examples = len(examples)
    buckets = [ [] for _ in range(values[0].shape[0])]
    for i,value in enumerate(values):
        buc = np.argmax(value)
        buckets[buc].append(i)
    for i in range(len(buckets)):
        buckets[i].sort(key=lambda x: values[x][i],reverse=True)
        
    # print(buckets)
    # for x in buckets:
    #     print([values[i] for i in x])
    ans = sum(values)
    order = []
    # print(ans)
    p=0
    for _ in range(num_examples):
        while(len(buckets[p])==0):
            p+=1
            if p==len(buckets):
                p=0
        order.append(buckets[p][0])
        buckets[p].pop(0)
        p+=1
        if p==len(buckets):
            p=0

    result=[]
    for i in order:
        result.append(examples[i])
    return result

def select_order_4shot(examples,values):
    assert len(examples) == len(values)
    num_examples = len(examples)
    buckets = [ [] for _ in range(values[0].shape[0])]
    for i,value in enumerate(values):
        buc = np.argmax(value)
        buckets[buc].append(i)
    for i in range(len(buckets)):
        buckets[i].sort(key=lambda x: values[x][i],reverse=True)
        
    # print(buckets)
    # for x in buckets:
    #     print([values[i] for i in x])
    ans = sum(values)
    order = []
    # print(ans)
    if len(buckets[0])==3:
        for i in range(3):
            order.append(buckets[0][i])
        order.append(buckets[1][0])
    elif len(buckets[1])==3:
        for i in range(3):
            order.append(buckets[1][i])
        order.append(buckets[0][0])      
    else:
        p=0
        for _ in range(num_examples):
            while(len(buckets[p])==0):
                p+=1
                if p==len(buckets):
                    p=0
            order.append(buckets[p][0])
            buckets[p].pop(0)       
            p+=1
            if p==len(buckets):
                p=0
    result=[]
    for i in order:
        result.append(examples[i])
    return result

def select_order_by_insert(examples,values,alpha):
    assert len(examples) == len(values)
    num_examples = len(examples)
    
    def cal_score(values,tmp,alpha):
        num = len(tmp)
        weights = [alpha**i - alpha**(i+1) for i in range(num-1)]
        weights.append(alpha**(num-1))
        weights = weights[::-1]
        ans=sum([weights[i]*values[tmp[i]] for i in range(num)])
        entropy = sum([-ans[i]*math.log(ans[i]) for i in range(ans.shape[0])])
        return entropy
           
    order=[0]
    for i in range(1,num_examples):
        scores = []
        for pos in range(0,i+1):
            order.insert(pos,i)
            scores.append(cal_score(values,order,alpha))
            del order[pos]
        pos = np.array(scores).argmax()
        order.insert(pos,i)
    
    result=[]
    for i in order:
        result.append(examples[i])
    return result


@torch.no_grad()
def get_demo_bias(model,device,tokenizer,dataset_train,demo_str):
    empty_input_str, answer_str = dataset_train.empty_example()
    inputs = tokenizer(
        demo_str+empty_input_str,
        return_tensors='pt',
    )
    answer_encoding = tokenizer(
        answer_str,
        padding=True,
        return_tensors='pt',
    ).to(device)
    if answer_encoding.input_ids.shape[1] == 1: # classification

        with torch.autocast(device_type="cuda"):
            logits = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device)
                ).logits
            
        logits = logits[0][-1]
        return torch.softmax(logits[answer_encoding.input_ids.flatten()],dim=0).cpu().numpy() 
        
    else: #
        raise NotImplementedError("")
    
def entropy(vec):
    return -np.sum(vec*np.log(vec))

def greedy_order(model,device,tokenizer,dataset_train,examples):

    order = []
    num = len(examples)
    examples = copy.deepcopy(examples)
    
    for _ in range(num):
        entropys = []
        for index in examples:
            demo_str = dataset_train.get_demo_from_indices([index]+order)
            entropys.append( entropy(get_demo_bias(model,device,tokenizer,dataset_train,demo_str)) )
        index = np.argmax(np.array(entropys))
        order.insert(0,examples[index])
        del examples[index]
    
    return order            
        
def greedy_insert_order(model,device,tokenizer,dataset_train,examples):
    order = []
    num = len(examples)
    
    for i in range(num):
        entropys = []
        for j in range(i+1):
            demo_str = dataset_train.get_demo_from_indices(order[:j]+[examples[i]]+order[j:])
            entropys.append( entropy(get_demo_bias(model,device,tokenizer,dataset_train,demo_str)) )
        arg_j = np.argmax(np.array(entropys))
        order = order[:arg_j]+[examples[i]]+order[arg_j:]
    
    return order
    
    
    
        
    
    
    
     