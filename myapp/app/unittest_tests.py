import pickle
from main import clf
import unittest

class Test_TestAccuracy(unittest.TestCase):
    def test_accuracy(self):

        # Load test data
        with open("data/test_data.pkl", "rb") as file:
            test_data = pickle.load(file)

        # Unpack the tuple
        X_test, y_test = test_data

        # Compute accuracy of classifier
        acc = clf.score(X_test, y_test)

        # Accuracy should be over 90%
        self.assertGreater(acc,0.9) 

if __name__ =='__main__':
    unittest.main()