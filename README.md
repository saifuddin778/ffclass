ffclass
=======

Classifier for uniformly distributed classes - classifies based on comparison of the results of multiple decision objects.

###Usage:
To build a forest of classifying object:
```python
>>> from ffclass import forest_fclass
>>> data = [[f1, f2 ... fn, class], ..., [f1, f2, ... fn, class]]
>>> ff = forest_fclass(data)
>>> ff.predict([f1,f2, ... , fn])
>>> "predicted class"
```
To utilize just a single decision object:
```python
>>> from ffclass import ffclass
>>> data = [[f1, f2 ... fn, class], ..., [f1, f2, ... fn, class]]
>>> ff = ffclass(data)
>>> ff.query([f1,f2, ... , fn])
>>> "predicted class"
```
#####Some Details:
1. Shuffle the data.
2. Distribute into chunks.
3. Build a classification tree for each of the chunks.
4. Classify the input by taking the winner from all the classifiers.

#####Possible Improvements (TODOS):
1. Initiate with a completely uniform shuffling.
2. Calculate joint-probability for each of the features i.e. `P(f1 | f2,f3,..fn)` and take it into the account while predicting.

