from enum import Enum
from typing import List
from process_data import load_corrected_data


class Topic(str, Enum):
    ABORTION = "Legalization of Abortion"
    ATHEISM = "Atheism"
    CLIMATE = "Climate Change"
    FEMINISM = "Feminist Movement"
    HRC = "Hillary Clinton"


class Stance(str, Enum):
    AGAINST = "AGAINST"
    FAVOR = "FAVOR"
    NONE = "NONE"


class Dataset(Enum):
    TRAIN = 0
    TEST = (1,)
    ALL = 2


# class Dataset(Enum):
#     TEST = 0,
#     TRAIN = 1,
#     FULL = 2


def get_abortion_indices() -> List[int]:
    return get_topic_indices(Topic.ABORTION)


def get_atheism_indices() -> List[int]:
    return get_topic_indices(Topic.ATHEISM)


def get_climate_indices() -> List[int]:
    return get_topic_indices(Topic.CLIMATE)


def get_feminism_indices() -> List[int]:
    return get_topic_indices(Topic.FEMINISM)


def get_hrc_indices() -> List[int]:
    return get_topic_indices(Topic.HRC)


def get_topic_indices(topic: Topic) -> List[int]:
    data = load_data()
    return [i for i in range(len(data)) if data[i]["topic"] == topic]


def load_data() -> List:
    from enum import Enum


class Topic(str, Enum):
    ABORTION = "Legalization of Abortion"
    ATHEISM = "Atheism"
    CLIMATE = "Climate Change"
    FEMINISM = "Feminist Movement"
    HRC = "Hillary Clinton"


class Stance(str, Enum):
    AGAINST = "AGAINST"
    FAVOR = "FAVOR"
    NONE = "NONE"


def get_abortion_indices(data) -> List[int]:
    return get_topic_indices(Topic.ABORTION, data)


def get_atheism_indices(data) -> List[int]:
    return get_topic_indices(Topic.ATHEISM, data)


def get_climate_indices(data) -> List[int]:
    return get_topic_indices(Topic.CLIMATE, data)


def get_feminism_indices(data) -> List[int]:
    return get_topic_indices(Topic.FEMINISM, data)


def get_hrc_indices(data) -> List[int]:
    return get_topic_indices(Topic.HRC, data)


def get_topic_indices(topic: Topic, data) -> List[int]:
    return [i for i in range(len(data)) if data[i]["topic"] == topic]


def load_data(dataset: Dataset) -> List:
    if dataset == Dataset.ALL:
        return load_corrected_data("data/semeval2016_corrected.txt")
    elif dataset == Dataset.TEST:
        return load_corrected_data("data/semeval2016_corrected_test.csv")
    elif dataset == Dataset.TRAIN:
        return load_corrected_data("data/semeval2016_corrected_train.csv")
    else:
        assert 1 == 2
