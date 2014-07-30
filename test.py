from datasets import load_seeds
from methods_ import Functions
from ffclass import forest_fclass
funcs_ = Functions()

x, y = load_seeds()
data = map(lambda n: funcs_.flatten(n, []), zip(x,y))
print len(data)
def test_ffclass(data):
    ff = forest_fclass(data)
    test = {'false': 0, 'true': 0}
    for a in data:
        pred = ff.predict(a[:-1])
        real = a[-1]
        if real == pred:
            test['true'] += 1
        else:
            test['false'] += 1
    return test

print test_ffclass(data)
