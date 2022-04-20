from enum import Enum
from process_data import load_corrected_data

class Topic(str, Enum):
    ABORTION = "Legalization of Abortion"
    ATHEISM = "Atheism"
    CLIMATE = "Climate Change"
    FEMINISM = "Feminist Movement"
    HRC = "Hillary Clinton"


def load_abortion_tweets():
    return load_topic_data(Topic.ABORTION)
def load_atheism_tweets():
    return load_topic_data(Topic.ATHEISM)
def load_climate_tweets():
    return load_topic_data(Topic.CLIMATE)
def load_feminism_tweets():
    return load_topic_data(Topic.FEMINISM)
def load_hrc_tweets():
    return load_topic_data(Topic.HRC)

def load_topic_data(topic: Topic):
    return list(filter(lambda dict: dict['topic'] == topic, load_data()))

def load_data():
    return load_corrected_data('data/semeval2016_corrected.txt')
    
print(load_hrc_tweets());
