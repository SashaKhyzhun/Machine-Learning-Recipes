from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn


def main():
    # Load data set
    iris = learn.datasets.load_dataset('iris')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Build 3 layers DNN with 10, 20, 10 units respectively
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    prediction = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, prediction)

    print('Accuracy: {0:f}'.format(score))


main()