# Testing CI/CD pipelines with GitHub Actions


## What is GH Actions?

This is an amazing tool that allows you to define automatic workflows for specific events within a GH repo. For instance you can create Python script versions of all your Jupyter notebooks every time you push or pull changes from the remote repository.

In this lab you will set up an action that will run the unit tests defined for your code every time you push changes to the remote repo.

To give you an idea of the flexibility of this tool, you could also (although not covered in this lab) set the action to build a Docker image out of your code if all the unit tests were passed and sent that image to a Google Cloud Bucket where it can be used to deploy your code. This would mean that you successfully automated your deployment with every push of changes.


## Testing the CI/CD pipeline

Within the `app` directory a copy of the server that serves predictions for the Wine dataset (that you used in a previous ungraded lab) is provided. The file is the same as in that previous lab with the exception that the classifier is loaded directly into global state instead of within a function that runs when the server is started. This is done because you will be performing unit tests on the classifier without starting the server.

### Unit testing with pytest

To perform unit testing you will use the `pytest` library. When using this library you should place your tests within a Python script that starts with the prefix `test_`, in this case it is called `test_clf.py` as you will be testing the classifier. 

Let's take a look at the contents of this file:

```python
import pickle
from main import clf

def test_accuracy():

    # Load test data
    with open("data/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)

    # Unpack the tuple
    X_test, y_test = test_data

    # Compute accuracy of classifier
    acc = clf.score(X_test, y_test)

    # Accuracy should be over 90%
    assert acc > 0.9
```

There is only one unit test defined in the `test_accuracy` function. This function loads the test data that was saved in pickle format and is located in the `data/test_data.pkl` file. Then it uses this data to compute the accuracy of the classifier on this test data. Something important is that this data is **not scaled** as the test expects the classifier to be a `sklearn.pipeline.Pipeline` which first step is a `sklearn.preprocessing.StandardScaler`.

If the accuracy is greater than 90% then the test passes. Otherwise it fails.

## Running the GitHub Action

To run the unit test using the CI/CD pipeline you need to push some changes to the remote repository. To do this, **add a comment somewhere in the `main.py` file and save the changes**.

Now you will use git to push changes to the remote version of your fork. 
- Begin by checking that there was a change using the `git status` command. You should see `main.py` in the list that is outputted.

- Now stage all of the changes by using the command `git add --all`.
- Create a commit with the command `git commit -m "Testing the CI/CD pipeline"`. 
- Finally push the changes using the command `git push origin main`.

With the push the CI/CD pipeline should have been triggered. To see it in action visit your forked repo in a browser and click the `Actions` button.



## Running the pipeline more times

### Changing the code

Suppose a teammate tells you that the Data Science team has developed a new model with an accuracy of 95% (the current one has 91%) so you decide to use new model instead. It is found in the `models/wine-95.pkl` file so to use it in your webserver you need to modify `main.py`. You should change the following lines:

```python
with open("models/wine.pkl", "rb") as file:
    clf = pickle.load(file)
```

So they look like this:

```python
with open("models/wine-95.pkl", "rb") as file:
    clf = pickle.load(file)
```

Once the change is saved, use git to push the changes as before. Use the following commands in sequence:

- `git add -all`
- `git commit -m "Adding new classifier"`
- `git push origin main`

With the push the CI/CD pipeline should have been triggered again. Once again go into the browser and check it. This time you will find that the tests failed. This can be done by the red icon next to the run:

![bad-run](../../assets/bad-run.png)

So, what happened?
You can dig deeper by going into the job and then into the steps that made it up. You should see something like this:

![error-detail](../../assets/error-detail.png)

The unit test failed because this new model has an accuracy lower to 90%. This happened because due to some miscommunication between teams, the Data Science team did not provide a `sklearn.pipeline.Pipeline` which first step is a `sklearn.preprocessing.StandardScaler`, but only the model since they expected the test data to be already scaled.

### Changing the code again

With this in mind you ask them to provide the model with the required characteristics. This one  is found in the `models/wine-95-fixed.pkl` file so to use it  you need to modify `main.py` once again. You should change the following lines:

```python
with open("models/wine-95.pkl", "rb") as file:
    clf = pickle.load(file)
```

So they look like this:

```python
with open("models/wine-95-fixed.pkl", "rb") as file:
    clf = pickle.load(file)
```

You also decided to add a new unit test to catch this error explicitly if it happens again. To do so modify the `test_clf.py` file to include these imports:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

And add a new unit test that looks like this:

```python
def test_pipeline_and_scaler():

    # Check if clf is an instance of sklearn.pipeline.Pipeline 
    isPipeline = isinstance(clf, Pipeline)
    assert isPipeline
    
    if isPipeline:
        # Check if first step of pipeline is an instance of 
        # sklearn.preprocessing.StandardScaler
        firstStep = [v for v in clf.named_steps.values()][0]
        assert isinstance(firstStep, StandardScaler)
```

This new test will check that the classifier is of type `sklearn.pipeline.Pipeline` and that its first step is a `sklearn.preprocessing.StandardScaler`.

Once the change is saved, use git to push the changes as before. Use the following commands in sequence:

- `git add -all`
- `git commit -m "Adding new classifier with scaling"`
- `git push origin main`

Now all of the tests should pass! With this you can be sure that this new version of the model is working as expected.

-----
