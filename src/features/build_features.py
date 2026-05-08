"""Feature engineering pipeline for IEEE-CIS fraud detection."""

from __future__ import annotations
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROVIDER_MAP = {
    "gmail": "google",
    "googlemail": "google",
    "yahoo": "yahoo",
    "ymail": "yahoo",
    "rocketmail": "yahoo",
    "hotmail": "microsoft",
    "outlook": "microsoft",
    "msn": "microsoft",
    "live": "microsoft",
    "att": "att",
    "sbcglobal": "att",
    "bellsouth": "att",
    "icloud": "apple",
    "me": "apple",
    "mac": "apple",
    "comcast": "comcast",
    "aol": "aol",
    "anonymous": "anonymous",
    "missing": "missing",
}


RAW_V_DERIVED_PATTERN = re.compile(r"^V\d+($|_.*)")


def require_columns(X: pd.DataFrame, columns: list[str], owner: str) -> pd.DataFrame:
    missing = [col for col in columns if col not in X.columns]
    if missing:
        raise ValueError(
            f"{owner}: missing {len(missing)} required columns. "
            f"First missing: {missing[:10]}"
        )
    return X.loc[:, columns]


validate_columns = require_columns


def assert_unique_columns(columns, owner: str) -> None:
    duplicated = pd.Index(columns)[pd.Index(columns).duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{owner}: duplicate columns found: {duplicated[:10]}")


def assert_no_unprefixed_v_derivatives(columns, owner: str) -> None:
    bad = [col for col in columns if RAW_V_DERIVED_PATTERN.match(str(col))]
    if bad:
        raise ValueError(
            f"{owner}: unprefixed raw V-derived columns leaked: {bad[:10]}"
        )


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        require_columns(X, ["TransactionDT"], "DatetimeFeatures.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = require_columns(X, ["TransactionDT"], "DatetimeFeatures.transform")[
            "TransactionDT"
        ]

        X["hour_of_day"] = (dt // 3600) % 24
        X["day_of_week"] = (dt // 86400) % 7
        X["is_weekend"] = (X["day_of_week"] >= 5).astype("int8")
        X["time_since_start"] = dt

        return X


class AmountFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        require_columns(X, ["TransactionAmt"], "AmountFeatures.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        amount = require_columns(X, ["TransactionAmt"], "AmountFeatures.transform")[
            "TransactionAmt"
        ]

        X["amt_log"] = np.log1p(amount)
        X["amt_rounded"] = (amount % 1 == 0).astype("int8")
        X["amt_cents"] = amount % 1

        return X


class EmailFeatures(BaseEstimator, TransformerMixin):
    required_cols = ["P_emaildomain", "R_emaildomain"]

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y=None):
        X_checked = require_columns(X, self.required_cols, "EmailFeatures.fit")

        p = X_checked["P_emaildomain"].fillna("missing")
        r = X_checked["R_emaildomain"].fillna("missing")

        p_counts = p.value_counts(dropna=False)
        r_counts = r.value_counts(dropna=False)

        p_denom = len(p) + self.alpha * (len(p_counts) + 1)
        r_denom = len(r) + self.alpha * (len(r_counts) + 1)

        self.p_email_freq_ = (p_counts + self.alpha) / p_denom
        self.r_email_freq_ = (r_counts + self.alpha) / r_denom
        self.p_unseen_freq_ = self.alpha / p_denom
        self.r_unseen_freq_ = self.alpha / r_denom

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_checked = require_columns(X, self.required_cols, "EmailFeatures.transform")

        p = X_checked["P_emaildomain"].fillna("missing")
        r = X_checked["R_emaildomain"].fillna("missing")

        X["P_email_freq"] = p.map(self.p_email_freq_).fillna(self.p_unseen_freq_)
        X["R_email_freq"] = r.map(self.r_email_freq_).fillna(self.r_unseen_freq_)
        X["email_match"] = (p == r).astype("int8")

        p_provider_raw = p.str.split(".", n=1).str[0]
        X["P_email_provider"] = p_provider_raw.map(PROVIDER_MAP).fillna("other")

        return X


class CardAggFeatures(BaseEstimator, TransformerMixin):
    required_cols = ["card1", "TransactionAmt"]

    def fit(self, X: pd.DataFrame, y=None):
        X_checked = require_columns(X, self.required_cols, "CardAggFeatures.fit")

        stats = X_checked.groupby("card1")["TransactionAmt"].agg(
            ["mean", "std", "count"]
        )

        self.card1_mean_ = stats["mean"]
        self.card1_std_ = stats["std"].fillna(0)
        self.card1_count_ = stats["count"]

        self.global_mean_ = X_checked["TransactionAmt"].mean()
        self.global_std_ = X_checked["TransactionAmt"].std()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        require_columns(X, self.required_cols, "CardAggFeatures.transform")

        X["card1_amt_mean"] = X["card1"].map(self.card1_mean_).fillna(
            self.global_mean_
        )
        X["card1_amt_std"] = X["card1"].map(self.card1_std_).fillna(
            self.global_std_
        )
        X["card1_tx_count"] = X["card1"].map(self.card1_count_).fillna(1.0)

        return X


class MFeatures(BaseEstimator, TransformerMixin):
    binary_m = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
    required_cols = [f"M{i}" for i in range(1, 10)]

    def fit(self, X: pd.DataFrame, y=None):
        X_checked = require_columns(X, self.required_cols, "MFeatures.fit")

        cats = set(X_checked["M4"].dropna().unique())
        cats.add("missing")

        self.m4_categories_ = sorted(f"M4_{cat}" for cat in cats)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        require_columns(X, self.required_cols, "MFeatures.transform")

        X[self.binary_m] = X[self.binary_m].replace({"T": 1, "F": 0}).fillna(-1)

        m4_dummies = pd.get_dummies(
            X["M4"].fillna("missing"),
            prefix="M4",
            dtype="int8",
        )
        m4_dummies = m4_dummies.reindex(columns=self.m4_categories_, fill_value=0)

        X["M_sum_true"] = (X[self.binary_m] == 1).sum(axis=1)
        X["M_sum_false"] = (X[self.binary_m] == 0).sum(axis=1)

        return pd.concat([X.drop(columns=["M4"]), m4_dummies], axis=1)


class DFeatures(BaseEstimator, TransformerMixin):
    d_cols = [f"D{i}" for i in range(1, 16)]

    def fit(self, X: pd.DataFrame, y=None):
        require_columns(X, self.d_cols, "DFeatures.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        require_columns(X, self.d_cols, "DFeatures.transform")

        for col in self.d_cols:
            X[f"{col}_missing"] = X[col].isna().astype("int8")
            X[col] = X[col].fillna(-1)

        return X


class IdentityFeatures(BaseEstimator, TransformerMixin):
    id_cols = [f"id_{i:02d}" for i in range(1, 39)]

    def fit(self, X: pd.DataFrame, y=None):
        require_columns(X, self.id_cols, "IdentityFeatures.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        require_columns(X, self.id_cols, "IdentityFeatures.transform")

        X["has_identity"] = X[self.id_cols].notna().any(axis=1).astype("int8")

        for col in self.id_cols:
            X[col] = X[col].fillna(-999)

        return X


class VRawImputedRepresentation(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        X_checked = require_columns(X, self.columns_, "VRawImputedRepresentation.fit")

        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(X_checked)

        self.feature_names_out_ = (
            [f"{col}_value" for col in self.columns_]
            + [f"{col}_missing" for col in self.columns_]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_checked = require_columns(
            X,
            self.columns_,
            "VRawImputedRepresentation.transform",
        )

        values = self.imputer_.transform(X_checked)
        missing = X_checked.isna().astype("int8").values
        output = np.hstack([values, missing])

        return pd.DataFrame(
            output,
            columns=self.feature_names_out_,
            index=X_checked.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)


class VPCARepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        X_checked = require_columns(X, self.columns_, "VPCARepresentation.fit")

        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )

        X_imputed = self.imputer_.fit_transform(X_checked)
        X_scaled = self.scaler_.fit_transform(X_imputed)

        self.pca_.fit(X_scaled)

        self.n_pca_components_ = self.pca_.n_components_
        self.feature_names_out_ = (
            [f"v_pca_{i + 1:03d}" for i in range(self.n_pca_components_)]
            + [f"{col}_missing" for col in self.columns_]
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_checked = require_columns(X, self.columns_, "VPCARepresentation.transform")

        missing = X_checked.isna().astype("int8").values

        X_imputed = self.imputer_.transform(X_checked)
        X_scaled = self.scaler_.transform(X_imputed)
        X_pca = self.pca_.transform(X_scaled)

        output = np.hstack([X_pca, missing])

        return pd.DataFrame(
            output,
            columns=self.feature_names_out_,
            index=X_checked.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)


class StableCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_onehot_cardinality=20, alpha=1.0):
        self.max_onehot_cardinality = max_onehot_cardinality
        self.alpha = alpha
        self.missing_token = "__MISSING__"
        self.other_token = "__OTHER__"

    def _clean_values(self, series: pd.Series) -> pd.Series:
        return series.astype("string").fillna(self.missing_token).astype(str)

    def fit(self, X: pd.DataFrame, y=None):
        self.input_columns_ = list(X.columns)
        assert_unique_columns(self.input_columns_, "StableCategoricalEncoder.fit")

        self.categorical_cols_ = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        self.numeric_cols_ = [
            col for col in self.input_columns_
            if col not in self.categorical_cols_
        ]

        self.onehot_categories_ = {}
        self.onehot_feature_names_ = {}
        self.onehot_index_ = {}
        self.onehot_lookup_ = {}

        self.freq_maps_ = {}
        self.unseen_freq_ = {}

        for col in self.categorical_cols_:
            values = self._clean_values(X[col])
            counts = values.value_counts(dropna=False)

            if len(counts) <= self.max_onehot_cardinality:
                categories = counts.index.tolist()

                for token in [self.missing_token, self.other_token]:
                    if token not in categories:
                        categories.append(token)

                feature_names = [
                    f"{col}__onehot_{i:03d}"
                    for i in range(len(categories))
                ]

                self.onehot_categories_[col] = categories
                self.onehot_feature_names_[col] = feature_names
                self.onehot_index_[col] = {
                    category: i for i, category in enumerate(categories)
                }
                self.onehot_lookup_[col] = dict(zip(feature_names, categories))

            else:
                denom = len(values) + self.alpha * (len(counts) + 1)
                self.freq_maps_[col] = (counts + self.alpha) / denom
                self.unseen_freq_[col] = self.alpha / denom

        self.feature_names_out_ = list(self.numeric_cols_)

        for col in self.categorical_cols_:
            if col in self.onehot_feature_names_:
                self.feature_names_out_.extend(self.onehot_feature_names_[col])
            else:
                self.feature_names_out_.append(f"{col}__freq")

        assert_unique_columns(
            self.feature_names_out_,
            "StableCategoricalEncoder.feature_names_out_",
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = require_columns(
            X,
            self.input_columns_,
            "StableCategoricalEncoder.transform",
        )

        parts = [X.loc[:, self.numeric_cols_].copy()]

        for col in self.categorical_cols_:
            values = self._clean_values(X[col])

            if col in self.onehot_categories_:
                categories = self.onehot_categories_[col]
                known_categories = set(categories) - {self.other_token}

                values = values.where(values.isin(known_categories), self.other_token)
                codes = values.map(self.onehot_index_[col]).astype("int64").to_numpy()

                arr = np.zeros((len(X), len(categories)), dtype="int8")
                arr[np.arange(len(X)), codes] = 1

                parts.append(
                    pd.DataFrame(
                        arr,
                        columns=self.onehot_feature_names_[col],
                        index=X.index,
                    )
                )

            else:
                encoded = values.map(self.freq_maps_[col]).fillna(
                    self.unseen_freq_[col]
                )

                parts.append(
                    pd.DataFrame(
                        {f"{col}__freq": encoded.astype("float32")},
                        index=X.index,
                    )
                )

        output = pd.concat(parts, axis=1)

        return output.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)


class NumericFinalizer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        assert_unique_columns(X.columns, "NumericFinalizer.fit")

        non_numeric = [
            col for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]

        if non_numeric:
            raise TypeError(f"Non-numeric columns remain: {non_numeric[:20]}")

        self.columns_ = list(X.columns)

        clean = X.replace([np.inf, -np.inf], np.nan)
        self.medians_ = clean.median(numeric_only=True).fillna(0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = require_columns(X, self.columns_, "NumericFinalizer.transform").copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.medians_)

        return X.loc[:, self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


class FraudFeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, v_groups, v_strategy, max_onehot_cardinality=20):
        self.v_groups = v_groups
        self.v_strategy = v_strategy
        self.max_onehot_cardinality = max_onehot_cardinality

    def _lookup_v_strategy(self, missing_rate):
        if missing_rate in self.v_strategy:
            return self.v_strategy[missing_rate]["strategy"]

        for key, value in self.v_strategy.items():
            if np.isclose(float(key), float(missing_rate)):
                return value["strategy"]

        raise ValueError(f"No V strategy found for missing rate {missing_rate}")

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeaturePipeline expects a pandas DataFrame.")

        assert_unique_columns(X.columns, "FraudFeaturePipeline.fit")

        self.input_columns_ = list(X.columns)
        X_raw = X.copy()

        self.v_source_cols_ = sorted(
            {col for cols in self.v_groups.values() for col in cols}
        )

        self.datetime_ = DatetimeFeatures().fit(X_raw, y)
        self.amount_ = AmountFeatures().fit(X_raw, y)
        self.email_ = EmailFeatures().fit(X_raw, y)
        self.card_ = CardAggFeatures().fit(X_raw, y)
        self.m_ = MFeatures().fit(X_raw, y)
        self.d_ = DFeatures().fit(X_raw, y)
        self.identity_ = IdentityFeatures().fit(X_raw, y)

        self.feature_blocks_ = [
            ("datetime", self.datetime_, ["TransactionDT"]),
            ("amount", self.amount_, ["TransactionAmt"]),
            ("email", self.email_, ["P_emaildomain", "R_emaildomain"]),
            ("card", self.card_, ["card1", "TransactionAmt"]),
            ("m", self.m_, [f"M{i}" for i in range(1, 10)]),
            ("d", self.d_, [f"D{i}" for i in range(1, 16)]),
            ("identity", self.identity_, [f"id_{i:02d}" for i in range(1, 39)]),
        ]

        self.v_representations_ = []

        for missing_rate, cols in sorted(self.v_groups.items()):
            strategy = self._lookup_v_strategy(missing_rate)

            if strategy == "raw":
                transformer = VRawImputedRepresentation()
            elif strategy == "pca":
                transformer = VPCARepresentation(n_components=0.95, random_state=42)
            else:
                raise ValueError(f"Unknown V strategy: {strategy}")

            source = require_columns(
                X_raw,
                cols,
                f"V group missing={missing_rate:.2f}",
            )

            transformer.fit(source, y)

            self.v_representations_.append(
                (missing_rate, list(cols), strategy, transformer)
            )

        engineered = self._engineer_features(X_raw)

        self.categorical_ = StableCategoricalEncoder(
            max_onehot_cardinality=self.max_onehot_cardinality
        ).fit(engineered, y)

        encoded = self.categorical_.transform(engineered)

        self.finalizer_ = NumericFinalizer().fit(encoded, y)
        self.feature_names_out_ = self.finalizer_.get_feature_names_out()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_raw = require_columns(
            X,
            self.input_columns_,
            "FraudFeaturePipeline.transform",
        ).copy()

        engineered = self._engineer_features(X_raw)
        encoded = self.categorical_.transform(engineered)
        finalized = self.finalizer_.transform(encoded)

        if finalized.isna().any().any():
            raise ValueError("Final feature table still contains NaNs.")

        assert_unique_columns(finalized.columns, "FraudFeaturePipeline.transform")

        return finalized

    def _apply_feature_blocks(self, X: pd.DataFrame) -> pd.DataFrame:
        X_work = X.copy()

        for name, transformer, required_cols in self.feature_blocks_:
            require_columns(X_work, required_cols, f"{name} block input")

            X_work = transformer.transform(X_work)

            if not isinstance(X_work, pd.DataFrame):
                raise TypeError(f"{name} block did not return a DataFrame.")

            assert_unique_columns(X_work.columns, f"{name} block output")

        return X_work

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_original = X.copy()
        X_work = self._apply_feature_blocks(X_original)

        main = X_work.drop(columns=self.v_source_cols_, errors="ignore")
        main = main.drop(columns=["TransactionID", "isFraud"], errors="ignore")

        assert_no_unprefixed_v_derivatives(main.columns, "main feature block")

        v_parts = []

        for missing_rate, cols, strategy, transformer in self.v_representations_:
            group_name = f"vgrp_{int(round(float(missing_rate) * 100)):02d}"

            source = require_columns(
                X_original,
                cols,
                f"V group missing={missing_rate:.2f}",
            )

            part = transformer.transform(source)
            part = part.add_prefix(f"{group_name}__")
            v_parts.append(part)

        output = pd.concat([main] + v_parts, axis=1)

        assert_unique_columns(output.columns, "engineered feature output")

        assert_no_unprefixed_v_derivatives(
            [col for col in output.columns if not str(col).startswith("vgrp_")],
            "engineered feature output",
        )

        return output

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)
