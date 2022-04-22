import re
from enum import Enum
from process_topic_data import Stance
from swear_words import swear_list
from process_topic_data import Topic

class Favor:
    ABORTION = ["rights", "my body", "will", "choice", "force", "woman/women", "rape"]
    ATHEISM = ["Evidence", "reason", "rational", "logic", "science", "equal", "free", "bigot", "scam"] + swear_list
    CLIMATE = ["#climate", "our", "warm", "stop", "plan"]
    FEMINISM = ["Harass"]
    HRC = ["Woman", "women", "rights"]

class Against:
    ABORTION = ["Life", "child", "murder", "kill","murder", "unborn"," die"]
    ATHEISM = ["[0-9]+:[0-9]+","#bible", "#holybible", "#holyspirit", "faith", "Christ", "Lord", "Jesus", "Bible", "God", "He", "Your", "His", "Father", "Amen," "pray"]
    CLIMATE = ["Scam", "hoax"]
    FEMINISM = ["#gamergate", "SJW", "oppress", "God"]
    HRC = ["Benghazi", "left", "lib", "email", "e-mail", "comm", "MSM", "found", "bill"]


def get_stance(tweet, cur_topic):
    ret = []
    if cur_topic == Topic.ATHEISM:
        ret = count_occurances(tweet, Favor.ABORTION, Against.ABORTION)
    elif cur_topic == Topic.HRC:
        ret = count_occurances(tweet, Favor.HRC, Against.HRC)
    elif cur_topic == Topic.CLIMATE:
        ret = count_occurances(tweet, Favor.CLIMATE, Against.CLIMATE)
    elif cur_topic == Topic.FEMINISM:
        ret = count_occurances(tweet, Favor.FEMINISM, Against.FEMINISM)
    else:
        ret = count_occurances(tweet, Favor.ATHEISM, Against.ATHEISM)

    if ret[0] > ret[1]:
        return Stance.FAVOR
    elif ret[0] < ret[1]:
        return Stance.AGAINST
    else:
        return Stance.NONE

def count_occurances(tweet, favor_list, against_list):
    for_count = 0
    against_count = 0
    for f_term in favor_list:
        if re.search(f_term, tweet, re.IGNORECASE) is not None:
            for_count += 1
    for if_term in against_list:
        if re.search(if_term, tweet, re.IGNORECASE) is not None:
            against_count += 1
    return [for_count,against_count]
