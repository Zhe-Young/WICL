from .BaseTask import BaseTask

from .SST2 import SST2
from .RTE import RTE
from .AGNews import AGNews
from .Subj import Subj
from .TREC import TREC
from .DBPedia import DBPedia
from .MR import MR
from .CR import CR


dataset_dict = {

    'sst2': SST2,
    'rte': RTE,
    'agnews': AGNews,
    'subj': Subj,
    'trec': TREC,
    'dbpedia': DBPedia,
    'mr': MR,
    'cr': CR
}

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](*args, **kwargs)


