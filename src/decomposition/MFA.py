import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from prince import MFA, MCA
from RPCA import RPCA


class RMFA(MFA):

    def __init__(self, groups=None, normalize=True, n_components=2, n_iter=10,
                 copy=True, check_input=True, random_state=None, engine='auto'):
        super().__init__(
            n_components=n_components,
            n_iter=n_iter,
            groups=groups,
            normalize=normalize,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):

        if self.groups is None:
            raise ValueError("Groups have to be specified")

        if self.check_input:
            check_array(X, dtype=[str, np.number])

        X = super()._prepare_input(X)

        self._check_group_type_consistency(X)

        self.partial_factor_analysis_ = {}
        for name, cols in sorted(self.groups.items()):
            if self.all_nums_[name]:
                fa = RPCA(
                    n_components=self.n_components,
                    copy=self.copy,
                    random_state=self.random_state
                )
            else:
                fa = MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Replace this with something better later on
        super(type(self).__bases__[0], self).fit(super()._build_X_global(X))

    def _check_group_type_consistency(self, X):
        self.all_nums_ = {}
        for name, cols in sorted(self.groups.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
            self.all_nums_[name] = all_num

    def transform(self, X):
        return super().transform(X)
