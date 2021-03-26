from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

from models.row2json import Row2Json
from models.train_test_validation import TrainTestAndValidation
from models.wine import Wine
from models.wine2vec import Wine2Vec
from models.wine_classifier import WineClassifier, emb_weights
from models.wine_corpus import WineCorpus
from models.wine_data_getter import WineMLDataGetter
from models.wine_dataset import WineDataSet
from models.wine_dict import WineDict
from models.wine_recommender import WineRecommender


def main():
    wine_recommender = WineRecommender()
    wine_recommender.load_corpus('corpus/all_wines.txt')

    iter = 30
    min_count = 1
    size = 300
    window = wine_recommender.wine_corpus.max_len
    file_path = f'wine2vec_pretrained/wine2vec_model_i{iter}_mc{min_count}_s{size}_w{window}'

    # wine_recommender.build_item2vec(
    #     iter=iter,
    #     min_count=min_count,
    #     size=size,
    #     window=window,
    #     save_path=file_path
    # )
    wine_recommender.load_item2vec(file_path)
    print(wine_recommender.recommend_by_description('oak coffee vanilla'))
    # wine_data_getter = WineMLDataGetter(
    #     wine_dataset=wine_recommender.wine_dataset,
    #     max_len=wine_recommender.wine_corpus.max_len,
    #     topn_varieties=7
    # )
    #
    # train_test_and_validation = TrainTestAndValidation(
    #     wine_data_getter.X,
    #     wine_data_getter.Y,
    #     test_size=0.2,
    #     valid_size=0.2
    # )
    #
    # vocabulary_size = len(wine_data_getter.word2index) + 1
    # embedding_size = 300
    #
    # wine_classifier = WineClassifier(
    #     vocabulary_size=vocabulary_size,
    #     embedding_size=embedding_size,
    #     max_seq_length=wine_recommender.wine_corpus.max_len,
    #     embedding_weights=emb_weights(
    #         word2vec=wine_recommender.wine2vec.model,
    #         word2index=wine_data_getter.word2index,
    #         vocabulary_size=vocabulary_size,
    #         embedding_size=embedding_size
    #     ),
    #     num_classes=len(set(train_test_and_validation.Y_test))
    # )
    wine_classifier.fit(
        X_train=train_test_and_validation.X_train,
        y_train=train_test_and_validation.Y_train,
        epochs=40,
        batch_size=100,
        X_validation=train_test_and_validation.X_validation,
        y_validation=train_test_and_validation.Y_validation
    )

    wine_classifier.save('model_lstm_test_40')
    # # wine_classifier.load('model_lstm_test_40')
    # wine_classifier.evaluate(train_test_and_validation.X_test, train_test_and_validation.Y_test)
    # Y_pred = wine_classifier.predict(train_test_and_validation.X_test)
    cm = confusion_matrix(
            y_true=train_test_and_validation.Y_test,
            y_pred=Y_pred
        )
    # print(
    #     cm
    # )
    #
    # print((cm / cm.astype(np.float32).sum(axis=1)) * 100)

    # print(wine_recommender.wine_dict[0])
    # print(wine_recommender.recommend(0))
    # print("---" * 10)
    # print(wine_recommender.wine_dict[1])
    # print(wine_recommender.recommend(wine_recommender.wine_dict[1].title))


if __name__ == '__main__':
    # wine130k = WineDataSet()
    # wine_dict = WineDict()
    # for row in tqdm(np.array(wine130k.data)):
    #     wine_dict.append(Wine(**Row2Json(row)))
    # del row, wine130k
    #
    # wine_corpus = WineCorpus(wine_dict)
    main()
    #
    # iter_ = 30
    # min_count = 1
    # size = 300
    # window = 80
    # file_path = f'wine2vec_pretrained/wine2vec_model_i{iter_}_mc{min_count}_s{size}_w{window}'
    #
    # wine2vec = Wine2Vec(
    #     sentences=wine_corpus,
    #     iter=iter_,
    #     min_count=min_count,
    #     size=size,
    #     window=window
    # )
    #
    # wine2vec.load(file_path)
    # wine2vec.most_similar('aromas')
