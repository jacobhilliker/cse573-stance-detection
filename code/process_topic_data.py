from enum import Enum
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


# class Dataset(Enum):
#     TEST = 0,
#     TRAIN = 1,
#     FULL = 2


def get_abortion_indices() -> list[int]:
    return get_topic_indices(Topic.ABORTION)


def get_atheism_indices() -> list[int]:
    return get_topic_indices(Topic.ATHEISM)


def get_climate_indices() -> list[int]:
    return get_topic_indices(Topic.CLIMATE)


def get_feminism_indices() -> list[int]:
    return get_topic_indices(Topic.FEMINISM)


def get_hrc_indices() -> list[int]:
    return get_topic_indices(Topic.HRC)


def get_topic_indices(topic: Topic) -> list[int]:
    data = load_data()
    return [i for i in range(len(data)) if data[i]["topic"] == topic]


def load_data() -> list:
    from enum import Enum
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


# class Dataset(Enum):
#     TEST = 0,
#     TRAIN = 1,
#     FULL = 2


def get_abortion_indices() -> list[int]:
    return get_topic_indices(Topic.ABORTION)


def get_atheism_indices() -> list[int]:
    return get_topic_indices(Topic.ATHEISM)


def get_climate_indices() -> list[int]:
    return get_topic_indices(Topic.CLIMATE)


def get_feminism_indices() -> list[int]:
    return get_topic_indices(Topic.FEMINISM)


def get_hrc_indices() -> list[int]:
    return get_topic_indices(Topic.HRC)


def get_topic_indices(topic: Topic) -> list[int]:
    data = load_data()
    return [i for i in range(len(data)) if data[i]["topic"] == topic]


def load_data() -> list:
    return load_corrected_data("data/semeval2016_corrected.txt")