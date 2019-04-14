from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

x = iris.data
y = iris.target

# test_size .5 means 50% of the data.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

my_classifier = KNeighborsClassifier()
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

print(str(predictions))
print(str(accuracy_score(y_test, predictions)))
