from datasets import load_dataset

from . import BaseTask

class SST2(BaseTask):
    def __init__(self, is_train=False,cache_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('glue', 'sst2',cache_dir=cache_dir)
        self.dataset = dataset['train'] if is_train else dataset['validation']
        self.class_num = 2
        self.preprocess_dataset()
        

    def templates_set_without_newline(self):
        return [
            ("Sentence: {text} Sentiment: ", " {answer}", ["negative", "positive"]),
            ("Input: {text} Prediction: ", " {answer}", ["negative", "positive"]),
            ("Review: {text} It was "," {answer}" ,["bad","good"]),
            ("Review: {text} Sentiment: ", " {answer}", ["bad", "good"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", example["sentence"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        
        return input_str, answer_str, label
    
    def empty_example(self):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{text}", "")
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        return input_str, answer_str
        
        
