from collections import Counter
from typing import Tuple, List

from tqdm import tqdm


def create_consecutive_relations(corpus: List[str],
                                 minimum_frequency: int = 2) -> List[Tuple[str, str]]:
    """
        Return a list of relations (edges of a graph) based on consecutive words.
        Each relation is a couple of word (w1, w2).
    """
    result = list()
    for doc in tqdm(corpus):
        words = doc.split()
        for i in range(len(words) - 1):
            result.append((words[i], words[i + 1]))
    return [k for k, v in Counter(result).items() if v >= minimum_frequency]


def create_context_relations(corpus: List[str],
                             window: int = 5,
                             minimum_frequency: int = 2) -> List[Tuple[str, str]]:
    """
        Return a list of relations (edges of a graph) based on a context window of specified size.
        Each relation is a couple of word (w1, w2).
    """
    result = list()
    for doc in tqdm(corpus):
        words = doc.split()
        size = len(words)
        for i in range(len(words)):
            for j in range(1, window + 1):
                if i - j > -1:
                    result.append((words[i], words[i - j]))
                if i + j < size:
                    result.append((words[i], words[i + j]))
    return [k for k, v in Counter(result).items() if v >= minimum_frequency]


# if __name__ == '__main__':
#     my_corpus = [
#         "ciao mamma guarda come mi diverto",
#         "ai lati di italia",
#         "i topi non avevano nipoti",
#         "bella zio come butta",
#         "ciao ciao arrivederci",
#     ]
#
#     print("\n > Consecutive relations:")
#     relations1 = create_consecutive_relations(corpus=my_corpus, minimum_frequency=1)
#     print(relations1)
#
#     print("\n > Context relations:")
#     relations2 = create_context_relations(corpus=my_corpus, window=5, minimum_frequency=1)
#     print(relations2)
