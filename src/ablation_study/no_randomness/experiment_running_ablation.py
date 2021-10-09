import os

import tensorflow as tf
from gensim.models import Word2Vec

from src.ablation_study.no_randomness.topic_modeling_ablation import GenerativeCooperativeTopicModelingAblation2
from tm_utils.business.topics_evaluation import get_topics_words_scores
from tm_utils.business.topics_utils import get_eng_keyed_vectors, save_scores_diz, save_html_topics_table
from tm_utils.config.configuration import TWENTY_NEWS_GROUP, IMDB_REVIEWERS, REUTERS, MODELS_DIR, ENG_STOPWORDS
from tm_utils.dao.dataset_dao import TwentyNewsgroupsDao, ImdbReviewsDao, ReutersNewswireDao

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__)

    print("\n\n >>> Settings")

    # ===== Set the input dataset, the number of topics and the number of words for cluster ===== #
    INPUT_DATASET = TWENTY_NEWS_GROUP
    print(f"\t\t input_dataset:  {INPUT_DATASET}")
    print("\n")

    NUM_TOP_WORDS = 50
    print(f"\t\t num. top words:  {NUM_TOP_WORDS}")
    print("\n\n")

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

    # Load FasText pretrained English embeddings
    print("\n >>> Load pretrained FasText english keyed vectors")
    eng_keyed_vectors = get_eng_keyed_vectors()

    # ===== Load trained embeddings model ===== #
    print("\n - Load trained word2vec model...")
    kv = Word2Vec.load(os.path.join(MODELS_DIR, INPUT_DATASET, "word2vec_final.model")).wv

    for NUM_TOPICS in (20,):
        print(f"\n\n #################### NUMBER OF TOPICS: {NUM_TOPICS} ####################")

        print("\n >>>>>>>>>>>>>>>>>>>> GCTM")
        for model_name in (
                # "CTM-e-no_randomness",
                "CTM-b-no_randomness",
        ):

            print(f"\n  ##### Model name: {model_name} ##### \n")

            if model_name == "CTM-e-no_randomness":
                gctm_obj = GenerativeCooperativeTopicModelingAblation2(num_topics=NUM_TOPICS,
                                                                       max_num_words=50,
                                                                       max_df=0.60,
                                                                       min_df=0.0005,
                                                                       stopwords=ENG_STOPWORDS,
                                                                       ngram_range=(1, 1),
                                                                       lowercase=True,
                                                                       max_features=None,
                                                                       num_epoches=5,
                                                                       batch_size=64,
                                                                       discr_learning_rate=0.005,
                                                                       random_seed_size=128,
                                                                       discriminator_hidden_dim=256,
                                                                       document_dim=None,
                                                                       embeddings=kv)
            else:
                gctm_obj = GenerativeCooperativeTopicModelingAblation2(num_topics=NUM_TOPICS,
                                                                       max_num_words=50,
                                                                       max_df=0.60,
                                                                       min_df=0.0005,
                                                                       stopwords=ENG_STOPWORDS,
                                                                       ngram_range=(1, 1),
                                                                       lowercase=True,
                                                                       max_features=None,
                                                                       num_epoches=5,
                                                                       batch_size=64,
                                                                       discr_learning_rate=0.005,
                                                                       random_seed_size=128,
                                                                       discriminator_hidden_dim=256,
                                                                       document_dim=None,
                                                                       embeddings=None)

            # ===== computation on corpus (dimensional reduction, clustering, summarization) ===== #
            print("\n >>> fit...")
            gctm_obj.fit(corpus=x_train_prep)

            # ====== get the extracted clusters of words ===== #
            print("\n >>>> get topics... \n")
            topics_matrix = gctm_obj.get_topics_words()
            [print("Topic", num_row + 1, ", size =", len(row), " | ", row)
             for num_row, row in enumerate(topics_matrix)]

            # ===== compute scores ===== #
            print("\n >>> Compute scores...")
            scores_diz = get_topics_words_scores(topics=topics_matrix,
                                                 kv=eng_keyed_vectors,
                                                 corpus=x_train_prep)
            print(scores_diz)
            print(round(scores_diz["quality_score"], 3))
            print(round(scores_diz["calinski_harabasz_score"], 3))
            print(round(scores_diz["silhouette_score"], 3))
            print(round(scores_diz["u_mass"], 3))
            print(round(scores_diz["c_uci"], 3))
            print(round(scores_diz["c_npmi"], 3))

            # ===== save topics and scores on FS ===== #
            print("\n >>> Save topics and scores on FS...")
            save_scores_diz(scores_diz=scores_diz,
                            name=model_name,
                            num_topics=NUM_TOPICS,
                            dataset_name=INPUT_DATASET)

            save_html_topics_table(topics_matrix=topics_matrix,
                                   name=model_name,
                                   num_topics=NUM_TOPICS,
                                   num_top_words=NUM_TOP_WORDS,
                                   dataset_name=INPUT_DATASET)

            print("\n ### END OF ITERATION! ### \n\n")
