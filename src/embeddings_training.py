from collections import Counter
from typing import Sequence
from typing import Tuple, List

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.poincare import PoincareModel
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


class PrintLossCallback(CallbackAny2Vec):
    """ Callback to print loss after each epoch """

    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = None

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print(f'Loss after epoch {self.epoch}: {loss:,}')
        else:
            print(f'Loss after epoch {self.epoch}: {loss - self.loss_previous_step:,}')
        self.epoch += 1
        self.loss_previous_step = loss


def train_poincare_embeddings(corpus: List[str],
                              strategy: str,
                              out_path: str):
    """
        Function for train Poincare (Hyperbolic) Embeddings and store them in keyed vectors format

        Args:
            corpus: corpus of texts
            strategy: 'consecutive' or 'context'
            out_path: path where store the obtained keyed vectors embeddings. The name must end with the '.kv' suffix.
    """
    if not out_path.endswith(".kv"):
        raise ValueError("The out name of the model must end with the '.kv' suffix!")
    if strategy == "consecutive":
        relations = create_consecutive_relations(corpus=corpus, minimum_frequency=5)
    elif strategy == "context":
        relations = create_context_relations(corpus=corpus, window=5, minimum_frequency=15)
    else:
        raise ValueError(f"Invalid strategy '{strategy}' (valid values: 'consecutive', 'context')!")
    print("Relations size", len(relations))
    relations = list(set(relations))
    print("Relations size", len(relations))
    model = PoincareModel(relations,
                          size=100,
                          alpha=0.1,
                          negative=5,
                          workers=1,
                          epsilon=1e-5,
                          regularization_coeff=1.0,
                          burn_in=10,
                          burn_in_alpha=0.01)
    batch_size = 128
    model.train(epochs=100, batch_size=batch_size, print_every=int(len(relations) / batch_size))
    model.kv.save_word2vec_format(out_path)


def train_word2vec_embeddings(corpus: List[str],
                              out_path: str,
                              min_count: int):
    """
        Function for train and store a Word2vec model starting form an input corpus

        Args:
            corpus: corpus of texts
            out_path: path where store the obtained word2vec model. The name must end with the '.model' suffix.
            min_count: minimum document frequency
    """
    if not out_path.endswith(".model"):
        raise ValueError("The out name of the model must end with the '.model' suffix!")
    model = Word2Vec(
        vector_size=100,
        alpha=0.025,
        window=5,
        min_count=min_count,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=8,
        min_alpha=0.00001,
        sg=0,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        cbow_mean=1,
        hashfxn=hash,
        epochs=100,
        null_word=0,
    )
    docs_tokenized = [doc.split() for doc in corpus]
    print("\n Build vocabulary...")
    model.build_vocab(docs_tokenized)
    print("corpus_count:", model.corpus_count)
    print("epochs:", model.epochs)
    print("corpus_total_words:", model.corpus_total_words)
    try:
        print("vocabulary size:", len(model.wv.vocab))
    except AttributeError:
        print("vocabulary size:", len(model.wv.key_to_index))
    print("vector_size:", model.vector_size)
    print("\n Start training...")
    model.train(docs_tokenized,
                total_examples=model.corpus_count,
                epochs=model.epochs,
                report_delay=1,
                compute_loss=True,
                callbacks=[PrintLossCallback()])
    model.callbacks = None
    # model.wv.save("out_name.kv"))
    model.save(out_path)
    print("training done! \n")


def try_embeddings_model(in_path: str, words: Sequence[str]):
    if in_path.endswith(".model"):
        kv = Word2Vec.load(in_path).wv
    elif in_path.endswith(".kv"):
        kv = KeyedVectors.load_word2vec_format(in_path)
    else:
        raise ValueError("the name of the model to test must end with these suffixes: ('.model', '.kv')!")
    print("\n >>> KV object fields:")
    print(kv.__dict__.keys())
    print("\n >>> most similar words:")
    for w in words:
        try:
            print(w, "-->", kv.most_similar(positive=[w], topn=30))
        except KeyError:
            print(f"'{w}' is not in vacabulary!")
