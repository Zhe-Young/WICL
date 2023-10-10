import argparse
import os
import json

import torch

#from models.bloom.modeling_bloom import BloomForCausalLM
#from bloom import BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from xglm import XGLMForCausalLM
#from transformers import XGLMForCausalLM

from dataset import get_dataset, dataset_dict

from utils.functional import select_past_key_value, setup_seed

from tqdm import tqdm

from itertools import product

import numpy as np

import copy


@torch.no_grad()
def calibration(model, dataset, tokenizer, device, past_key_values, past_attention_mask):

        
    input_str, output_str = dataset.empty_example()
    input_encoding = tokenizer(
        input_str,
        return_tensors='pt',
    ).input_ids.to(device)
    answer_encoding = tokenizer(
        output_str,
        padding=True,
        return_tensors='pt',
    ).to(device)
    
    # for i,token in enumerate(input_encoding.tolist()[0]):
    #     print(i,tokenizer.decode(token))
        
    if type(past_attention_mask)==list:
        all_logits_list = []
        for past_key_values_item,past_attention_mask_item in zip(past_key_values,past_attention_mask):

            if past_key_values_item is not None:
                attention_mask = torch.cat((past_attention_mask_item, torch.ones(input_encoding.shape, device=device)), dim=1)
            else:
                attention_mask = torch.ones(input_encoding.shape, device=device)
            
            if past_key_values_item is not None:
                past_key_values_cuda=()
                for kv_layer in past_key_values_item:
                    past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )
            else:
                    past_key_values_cuda=None

            with torch.autocast(device_type="cuda"):
                logits = model(
                    input_ids=input_encoding,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values_cuda
                    ).logits
                
            logits = logits[0][-1]
            all_logits = logits[answer_encoding.input_ids.flatten()]  
            all_logits = torch.softmax(all_logits,dim=-1)
            all_logits_list.append(all_logits) 
        return all_logits_list 
    
    else:
    
        if past_key_values is not None:
            attention_mask = torch.cat((past_attention_mask, torch.ones(input_encoding.shape, device=device)), dim=1)
        else:
            attention_mask = torch.ones(input_encoding.shape, device=device)
        
        if past_key_values is not None:
            past_key_values_cuda=()
            for kv_layer in past_key_values:
                past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )
        else:
                past_key_values_cuda=None

        with torch.autocast(device_type="cuda"):
            logits = model(
                input_ids=input_encoding,
                attention_mask=attention_mask,
                past_key_values=past_key_values_cuda
                ).logits
            
        logits = logits[0][-1]
        all_logits = logits[answer_encoding.input_ids.flatten()]  
        all_logits = torch.softmax(all_logits,dim=-1)   
     
    #bias = all_logits - torch.tensor([1/all_logits.shape[0] for _ in range(all_logits.shape[0])]).to(device)
    
    return all_logits

@torch.no_grad()
def validate(model, dataset, tokenizer, device, past_key_values, past_attention_mask , demos_offset=None,demos_scale=None,re_weight_place=None,calib=False):
    correct = 0
    total = 0
    
    if calib:
        calib_all_logits = calibration(model, dataset, tokenizer, device, past_key_values, past_attention_mask)
    

    for input_str, output_str, answer in dataset:
        input_encoding = tokenizer(
            input_str,
            return_tensors='pt',
        ).input_ids.to(device)
        answer_encoding = tokenizer(
            output_str,
            padding=True,
            return_tensors='pt',
        ).to(device)
        
        # for i,token in enumerate(input_encoding.tolist()[0]):
        #     print(i,tokenizer.decode(token))
            
        if answer_encoding.input_ids.shape[1] == 1: # classification
            if type(past_attention_mask)==list:
                #past_key_values_list,past_attention_mask_list = past_key_values,past_attention_mask
                all_logits = []
                
                for past_key_values_item,past_attention_mask_item in zip(past_key_values,past_attention_mask):
                    if past_key_values is not None:
                        attention_mask = torch.cat((past_attention_mask_item, torch.ones(input_encoding.shape, device=device)), dim=1)
                    else:
                        attention_mask = torch.ones(input_encoding.shape, device=device)
                        
                    if past_key_values_item is not None:
                        past_key_values_cuda=()
                        for kv_layer in past_key_values_item:
                            past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )
                    else:
                        past_key_values_cuda=None
            
                    with torch.autocast(device_type="cuda"):
                        logits = model(
                            input_ids=input_encoding,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values_cuda,
                            demos_offset=demos_offset,
                            demos_scale=demos_scale,
                            re_weight_place=re_weight_place,
                            ).logits
                        
                    logits = logits[0][-1]
                    logits = logits[answer_encoding.input_ids.flatten()]
                    all_logits.append(torch.softmax(logits,dim=-1))
                #all_logits = sum(all_logits)/len(all_logits) 
                #print(all_logits)
                    
            else:
                
                if past_key_values is not None:
                    attention_mask = torch.cat((past_attention_mask, torch.ones(input_encoding.shape, device=device)), dim=1)
                else:
                    attention_mask = torch.ones(input_encoding.shape, device=device)
                    
                if past_key_values is not None:
                    past_key_values_cuda=()
                    for kv_layer in past_key_values:
                        past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )
                else:
                    past_key_values_cuda=None
        
                with torch.autocast(device_type="cuda"):
                    logits = model(
                        input_ids=input_encoding,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values_cuda,
                        demos_offset=demos_offset,
                        demos_scale=demos_scale,
                        re_weight_place=re_weight_place
                        ).logits
                    
                logits = logits[0][-1]
                all_logits = logits[answer_encoding.input_ids.flatten()]
                all_logits = torch.softmax(all_logits,dim=-1)     
            
            
        else: # multi-choice 
            pass

        if calib:
            all_logits/=calib_all_logits
        preds = all_logits.argmax(dim=-1)
        correct += int(preds.item() == answer)
        total += 1
        
    acc = correct / total
    return acc

@torch.no_grad()
def get_past_kv_from_demos(model, device, demo_encoding_batch, attention_mask_batch):

    if demo_encoding_batch.shape[1] > 0:
        all_past_key_values = []
        for demo_encoding, attention_mask in zip(demo_encoding_batch, attention_mask_batch):
            with torch.autocast(device_type="cuda"):
                past_key_values = model(
                    input_ids=demo_encoding.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    use_cache=True
                ).past_key_values

            past_key_values_cpu = ()
            for layer_past in past_key_values:
                layer_past = tuple(past_state.cpu() for past_state in layer_past)
                past_key_values_cpu = past_key_values_cpu + (layer_past, )

            all_past_key_values.append(past_key_values_cpu)

        past_key_values = select_past_key_value(all_past_key_values)
    else: # zero-shot
        past_key_values = None
    
    return past_key_values

@torch.no_grad()
def get_negative_MSP(model,device,tokenizer,dataset_train,indices,demos_offset,demos_scale,re_weight_place,past_key_values,attention_mask_batch):
    self_pred_logits = 0
    for j,index in enumerate(indices):
        input_str, output_str, answer = dataset_train[index]
        #print(answer)
        input_encoding = tokenizer(
            input_str,
            return_tensors='pt',
        ).input_ids.to(device)
        answer_encoding = tokenizer(
            output_str,
            padding=True,
            return_tensors='pt',
        ).to(device)
        
        past_attention_mask = attention_mask_batch.clone()

        past_attention_mask[:,demos_offset[j][1]-1] = 0

        attention_mask = torch.cat((past_attention_mask, torch.ones(input_encoding.shape, device=device)), dim=1)

        past_key_values_cuda=()
        for kv_layer in past_key_values:
            past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )

        
        with torch.autocast(device_type="cuda"):
            out = model(
                input_ids=input_encoding,
                attention_mask=attention_mask,
                past_key_values=past_key_values_cuda,
                demos_offset=demos_offset,
                demos_scale=demos_scale,
                re_weight_place=re_weight_place
                )
        logits = out.logits[0][-1]
        all_logits = logits[answer_encoding.input_ids.flatten()] 
        tmp=torch.softmax(all_logits,dim=-1).cpu().numpy()
        self_pred_logits -= np.log(tmp[answer])
    return self_pred_logits/len(indices)

@torch.no_grad()
def get_neagtive_entropy(model,device,tokenizer,dataset_train,indices,demos_offset,demos_scale,re_weight_place,past_key_values,attention_mask_batch):
    input_str, output_str = dataset_train.empty_example()
    input_encoding = tokenizer(
        input_str,
        return_tensors='pt',
    ).input_ids.to(device)
    answer_encoding = tokenizer(
        output_str,
        padding=True,
        return_tensors='pt',
    ).to(device)
    
    past_attention_mask = attention_mask_batch.clone()

    attention_mask = torch.cat((past_attention_mask, torch.ones(input_encoding.shape, device=device)), dim=1)

    past_key_values_cuda=()
    for kv_layer in past_key_values:
        past_key_values_cuda+=( (kv_layer[0].to(device),kv_layer[1].to(device)), )

    
    with torch.autocast(device_type="cuda"):
        out = model(
            input_ids=input_encoding,
            attention_mask=attention_mask,
            past_key_values=past_key_values_cuda,
            demos_offset=demos_offset,
            demos_scale=demos_scale,
            re_weight_place=re_weight_place
            )
    
    logits = out.logits[0][-1]
    all_logits = logits[answer_encoding.input_ids.flatten()]  
    tmp = torch.softmax(all_logits,dim=-1).cpu().numpy() 
    return -np.sum(tmp*np.log(tmp))

@torch.no_grad()
def beam_search(model,device,tokenizer,dataset_train,indices,demos_offset,re_weight_place,past_key_values,attention_mask_batch,weight_space,indicator='MSP',beam_num=1,dataset_validate=None):
        
    sequences = [[1.0]*len(indices)]
    for i in range(len(indices)):
        tmp = []
        for sequence in sequences:
            for x in weight_space:
                demos_scale = copy.deepcopy(sequence)
                demos_scale[i]=x
                if indicator=='MSP':
                    score = get_negative_MSP(model,device,tokenizer,dataset_train,indices,demos_offset,demos_scale,re_weight_place,past_key_values,attention_mask_batch)
                elif dataset_validate is not None:
                    score = -1*validate(model, dataset_validate, tokenizer, device, past_key_values, attention_mask_batch, demos_offset,demos_scale,re_weight_place)
                else:
                    score = get_neagtive_entropy(model,device,tokenizer,dataset_train,indices,demos_offset,demos_scale,re_weight_place,past_key_values,attention_mask_batch)
                tmp.append((score,demos_scale))
        tmp.sort(key=lambda x:x[0])
        sequences = [tmp[j][1] for j in range(beam_num)]
    
    return sequences[0]

    
 

def main():
    parser = argparse.ArgumentParser()
    # Model setting
    parser.add_argument('--model', type=str)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument("--cache_dir",type=str,default='../.cache/huggingface/hub/',help="model cache directory")
    
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--strategy', type=str, default="truncate")
    parser.add_argument('--data_path', type=str, default="../.cache/huggingface/datasets/",help="data cache directory")
    parser.add_argument('--log_path', type=str)
    
    # Parameters
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--shot', type=int)
    parser.add_argument('--beam_num', type=int, default=1)
    
    # settings
    parser.add_argument('--calib', type=str, default="None")
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--indicator',type=str, default='MSP',help="Indicator for weight searching, it can be MSP or validate_100 or validate_xx")
    parser.add_argument('--re_weight_place',type=str, default="before_softmax" ,help="before_softmax or after_softmax")
    parser.add_argument('--weight_space',type=str,default="0.9 1.0 1.1")
    
    args = parser.parse_args()
    
    if os.path.exists(args.log_path):
        exit(0)
    else:
        print(vars(args))

    model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")


    model = XGLMForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,cache_dir=args.cache_dir)
    model = model.to(device)
    
    model.eval()
    print("Model initialized.")
    
    if args.task:
        dataset_list = [args.task]
    else:
        dataset_list = dataset_dict.keys()

    for dataset in dataset_list:
        dataset_train = get_dataset(dataset, is_train=True,cache_dir=args.data_path)
        dataset_val = get_dataset(dataset, is_train=False, max_data_num=2000,cache_dir=args.data_path)
        acc_list = []
        #demo_max_length = args.max_length - dataset_val.get_max_length(tokenizer)
        dataset_validate = None
        if args.indicator.startswith("validate"):
            setup_seed(0)
            _,num = args.indicator.split('_')
            num = int(num)
            _,_, indices = dataset_train.get_demos(tokenizer, shot=num ,balanced=True)
            dataset_validate = copy.deepcopy(dataset_train)
            dataset_validate.dataset = dataset_validate.dataset.select(indices)
            dataset_validate.examples = []
            dataset_validate.preprocess_dataset()

        for seed in tqdm(range(args.repeat_num)):
            setup_seed(seed)
            demo_encoding_batch, attention_mask_batch, indices = dataset_train.get_demos(tokenizer, shot=args.shot,balanced=args.balanced)
            
            demo_encoding_batch = []
            demos_offset=[]
            offset = 0
            for index in indices:
                #_,_,label = dataset_train[index]
                #print(label)
                
                s = dataset_train.get_demo_from_indices(index)
                encoded_s = tokenizer.encode(s)
                demos_offset.append((offset,offset+len(encoded_s)))
                offset+=len(encoded_s)
                demo_encoding_batch+=encoded_s
                
            attention_mask_batch = [1]*len(demo_encoding_batch)
            demos_scale = [1]*len(demos_offset)

            demo_encoding_batch = torch.LongTensor([demo_encoding_batch]).to(device)
            attention_mask_batch = torch.LongTensor([attention_mask_batch]).to(device)
            
            past_key_values = get_past_kv_from_demos(model,device,demo_encoding_batch,attention_mask_batch)
            
            icl_acc = validate(model, dataset_val, tokenizer, device, past_key_values, attention_mask_batch.view(1,-1) )

            attention_mask_batch = attention_mask_batch.view(1,-1)
            
               
            weight_space = list(map(float,args.weight_space.split()))
            #re_weight_place = 'before_softmax'
            demos_scale = beam_search(model,device,tokenizer,dataset_train,indices,demos_offset,args.re_weight_place,past_key_values,attention_mask_batch,weight_space,args.indicator,args.beam_num,dataset_validate)
            
            wicl_acc = validate(model, dataset_val, tokenizer, device, past_key_values, attention_mask_batch ,demos_offset,demos_scale,args.re_weight_place)                              

            
            acc_list.append({
                "ICL_acc": icl_acc,
                "WICL_acc": wicl_acc,
                "indices": indices,
                'demos_scale': demos_scale
            })
            #print(acc_list)
            #print(acc2)
    
        log_dict = {
            "args" : vars(args),
            "ICL_acc_avg": sum([item["ICL_acc"] for item in acc_list])/len(acc_list),
            "WICL_acc_avg": sum([item["WICL_acc"] for item in acc_list])/len(acc_list),
            "ICL_acc_max": max([item["ICL_acc"] for item in acc_list]),
            "WICL_acc_max": max([item["WICL_acc"] for item in acc_list]),
            "ICL_acc_std": np.std(np.array([item["ICL_acc"] for item in acc_list])),
            "WICL_acc_std": np.std(np.array([item["WICL_acc"] for item in acc_list])),
            "details": acc_list
        }
        
        if args.log_path:
            with open(args.log_path, 'w') as fp:
                fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
