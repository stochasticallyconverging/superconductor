import numpy as np
import pandas as pd
from sklearn.utils import check_array
from prince import MFA, MCA

from src.decomposition.RPCA import RPCA


class RFAMD(MFA):

    def __init__(self, normalize=True, hard_threshold=False, pca_n_components=2, mca_n_components=2, n_iter=10,
                 copy=True, check_input=True, random_state=None, engine='auto'):
        super().__init__(
            n_components=pca_n_components,
            n_iter=n_iter,
            normalize=normalize,
            copy=copy,
            check_input=check_input,
            random_state=random_state,
            engine=engine
        )
        if hard_threshold:
            raise NotImplementedError
        else:
            self.mca_n_components = mca_n_components
        self.groups = None

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):

        if self.check_input:
            check_array(X, dtype=[np.chararray, np.number])

        X = super()._prepare_input(X)

        self._one_group_per_variable_type(X)
        self._check_group_consistency(X)

        # numeric_data_aspect_ratio = self.groups['Numerical'].shape[1]/self.groups['Numerical'].shape[0]
        # categorical_data_aspect_ratio = self.groups['Categorical'].shape[1]/self.groups['Categorical'].shape[0]

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
                    n_components=self.mca_n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])

        # Replace this with something better later on
        super(type(self).__bases__[0], self).fit(super()._build_X_global(X))

    def _one_group_per_variable_type(self, X):
        num_cols = X.select_dtypes(np.number).columns.tolist()
        cat_cols = X.select_dtypes(np.chararray).columns.tolist()

        self.groups = {}
        if num_cols:
            self.groups['Numerical'] = num_cols
        else:
            raise ValueError('FAMD assumes that X has both categorical and numerical data. No numerical data.')

        if cat_cols:
            self.groups['Categorical'] = cat_cols
        else:
            raise ValueError('FAMD assumes that X has both categorical and numerical data. No categorical data.')

    def _check_group_consistency(self, X):
        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
            self.all_nums_[name] = all_num

    def transform(self, X):
        return super().transform(X)

