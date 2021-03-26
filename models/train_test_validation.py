from datetime import datetime

from sklearn.model_selection import train_test_split


class TrainTestAndValidation:
    def __init__(self, X, Y, test_size, valid_size):
        TEST_SIZE = test_size
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=TEST_SIZE,
            random_state=4,
            stratify=Y
        )
        VALID_SIZE = valid_size
        X_train, X_validation, Y_train, Y_validation = train_test_split(
            X_train,
            Y_train,
            test_size=VALID_SIZE,
            random_state=4,
            stratify=Y_train
        )
        print(f'{datetime.now()} - TRAINING DATA')
        print('Shape of input sequences: {}'.format(X_train.shape))
        print('Shape of output sequences: {}'.format(len(Y_train)))
        print("-" * 50)
        print(f'{datetime.now()} - VALIDATION DATA')
        print('Shape of input sequences: {}'.format(X_validation.shape))
        print('Shape of output sequences: {}'.format(len(Y_validation)))
        print("-" * 50)
        print(f'{datetime.now()} - TESTING DATA')
        print('Shape of input sequences: {}'.format(X_test.shape))
        print('Shape of output sequences: {}'.format(len(Y_test)))

        self._X_train, self._X_test, self._Y_train, self._Y_test = X_train, X_test, Y_train, Y_test
        self._X_validation, self._Y_validation = X_validation, Y_validation

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def X_validation(self):
        return self._X_validation

    @property
    def Y_train(self):
        return self._Y_train

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def Y_validation(self):
        return self._Y_validation
