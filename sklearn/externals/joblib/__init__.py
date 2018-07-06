import sys
import os

if os.environ.get('SKLEARN_SITE_JOBLIB', False):
    import joblib
    module = sys.modules['joblib']
else:
    import sklearn.externals._joblib
    module = sys.modules['sklearn.externals._joblib']

sys.modules[__name__] = module
