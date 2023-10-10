import random

import torch
from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, max_data_num=None, temp_index=0, demo="",channel=False):
        super().__init__()
        self.temp_index = temp_index
        self.examples = []
        self.max_data_num = max_data_num
        self.demo = demo
        self.templates = self.templates_set_without_newline()
        self.channel = channel
        
    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def preprocess_dataset(self):
        self.label_count = [0 for _ in range(self.class_num)]
        for example in self.dataset:
            example = self.preprocess_example(example)
            if example[0] is None:
                continue

            self.label_count[example[2]] += 1
            self.examples.append(example)
 
        if self.max_data_num is not None and self.max_data_num < len(self.examples): # truncate dataset
            random.seed(1)
            self.examples = random.sample(self.examples, self.max_data_num)


    def get_demo_from_indices(self, indices, labeled = None):
        demo_str = ""
        if isinstance(indices, int):
            indices = [indices]
        
        for index in indices:
            input_str, output_str, label = self.examples[index]
            if labeled!=None:
                if labeled[0]:
                    if self.channel:
                        demo_str += output_str[label] + input_str + " \n "
                    else:                
                        demo_str += input_str + output_str[label] + " \n "
                else:
                    demo_str += input_str +  " \n "
                del labeled[0]
            else:
                demo_str += input_str + output_str[label] + " \n "

        return demo_str

    def get_max_length(self, tokenizer):
        return max(len(tokenizer(
                [input_str +" " + candidate_str for candidate_str in output_str],
                padding=True
            ).input_ids[0]) for input_str, output_str, _ in self.examples)
        

    def get_demos(self, tokenizer, indices=None, shot = 8, balanced = False):
        
        if indices is None:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            if balanced:
                _ , output_str, _ = self.examples[0]
                class_num = len(output_str)
                max_num_per_bucket = shot//class_num +int(shot%class_num!=0)
                ans = 0
                buckets = [0 for _ in range(class_num)]
                tmp_indices = []
                for i in indices:
                    _,_,label=self.examples[i]
                    if buckets[label]>=max_num_per_bucket:
                        continue
                    buckets[label]+=1
                    tmp_indices.append(i)
                    if len(tmp_indices)>=shot:
                        break
                indices = tmp_indices
            else:
                
                indices = random.sample(indices, shot)
            
        demo_encoding_batch = []
        demo_encoding = []
        attention_mask_batch = []
        
        if type(indices)==int:
            indices = [indices]

        for index in indices:

            demo = self.get_demo_from_indices(index)
            demo_input_ids = tokenizer(demo).input_ids
            demo_encoding += demo_input_ids

        demo_encoding_batch.append(demo_encoding)
        attention_mask_batch.append([1] * len(demo_encoding))

        return demo_encoding_batch, attention_mask_batch, indices
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return self.demo + input_str, output_str, label

    def __iter__(self):
        for input_str, output_str, label in self.examples:
            yield self.demo + input_str, output_str, label

if __name__ == '__main__':
    pass
    