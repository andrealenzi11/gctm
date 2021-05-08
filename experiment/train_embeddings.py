import logging
import os

import math

from business.embeddings_training import train_word2vec_embeddings, try_embeddings_model
from config.configuration import TWENTY_NEWS_GROUP, MODELS_DIR, IMDB_REVIEWERS, REUTERS
from dao.dataset_dao import TwentyNewsgroupsDao, ImdbReviewsDao, ReutersNewswireDao
from gtcacs.text_compression2 import GTC_NN_LOGGER_NAME


if __name__ == '__main__':

    logger_gtcacs = logging.getLogger(GTC_NN_LOGGER_NAME)
    logger_gtcacs.setLevel(level=logging.INFO)
    logger_poincare = logging.getLogger("gensim.models.poincare")
    logger_poincare.setLevel(level=logging.INFO)

    INPUT_DATASET = TWENTY_NEWS_GROUP

    if INPUT_DATASET not in [TWENTY_NEWS_GROUP, IMDB_REVIEWERS, REUTERS]:
        raise ValueError(f"Invalid input dataset '{INPUT_DATASET}'!")

    if INPUT_DATASET == TWENTY_NEWS_GROUP:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            TwentyNewsgroupsDao().load_preprocessed()
    elif INPUT_DATASET == IMDB_REVIEWERS:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            ImdbReviewsDao().load_preprocessed()
    elif INPUT_DATASET == REUTERS:
        x_train, x_test, x_train_prep, x_test_prep, y_train_prep, y_test_prep = \
            ReutersNewswireDao().load_preprocessed()
    else:
        raise ValueError(f"Invalid input dataset '{INPUT_DATASET}'!")

    print(f"\n\n >>> Input dataset '{INPUT_DATASET}' info:")
    print("\t\t x_train size: ", len(x_train))
    print("\t\t x_test size: ", len(x_test))
    print("\t\t x_train_prep size: ", len(x_train_prep))
    print("\t\t x_test_prep size: ", len(x_test_prep))
    print("\t\t y_train_prep size: ", len(y_train_prep))
    print("\t\t y_test_prep size: ", len(y_test_prep))

    min_word_frequency_count = math.ceil(len(x_train_prep) * 0.0005)
    print(f"minimum word frequency count: {min_word_frequency_count}")

    MODEL_FILENAME = "word2vec_final.model"

    print(f"\n\n\n Train Word2Vec embeddings model for dataset '{INPUT_DATASET}'... \n")
    train_word2vec_embeddings(corpus=x_train_prep,
                              out_path=os.path.join(MODELS_DIR, INPUT_DATASET, MODEL_FILENAME),
                              min_count=min_word_frequency_count)

    print(f"\n\n\n Perform some trials with the trained Word2Vec embeddings model for dataset '{INPUT_DATASET}'... \n")
    try_embeddings_model(in_path=os.path.join(MODELS_DIR, INPUT_DATASET, MODEL_FILENAME),
                         words=[
                             "man", "difference", "power", "component", "catalogue",
                             "paul", "article", "subject", "pump", "university",
                             "hair", "girl", "obesity",
                         ])
