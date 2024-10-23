import pandas as pd
import arviz as az
import pytest
from pymer4 import Lmer
import os


@pytest.mark.usefixtures("sleepstudy", "sampledata")
class Test_Basic_Usage:
    MAX_CHAINS = os.cpu_count()
    EXPECTED_PRIOR_MEAN_SUMMARY_COLS = ["Estimate", "SD", "2.5_hdi", "97.5_hdi"]
    EXPECTED_PRIOR_MEDIAN_SUMMARY_COLS = ["Estimate", "MAD", "2.5_eti", "97.5_eti"]
    EXPECTED_MODEL_COMPARISON_COLS = [
        "ELPD_Diff",
        "ELPD_Diff_SE",
        "Eff_Params",
        "BF_elpd",
        "Sig_BF_elpd",
        "T-stat_elpd",
        "P-val_elpd",
    ]
    EXPECTED_POSTERIOR_MEAN_SUMMARY_COLS = [
        "Estimate",
        "SD",
        "2.5_hdi",
        "97.5_hdi",
        "Rhat",
        "BF_10",
        "Sig_BF_10",
        "P-val_hdi",
        "Sig_hdi",
    ]
    EXPECTED_POSTERIOR_MEAN_RFX_SUMMARY_COLS = [
        "Estimate",
        "SD",
        "2.5_hdi",
        "97.5_hdi",
        "Rhat",
    ]
    EXPECTED_POSTERIOR_MEDIAN_SUMMARY_COLS = [
        "Estimate",
        "MAD",
        "2.5_eti",
        "97.5_eti",
        "Rhat",
    ]
    EXPECTED_ARVIZ_INFERENCE_DATA_GROUPS = [
        "posterior",
        "posterior_predictive",
        "sample_stats",
        "prior",
        "prior_predictive",
        "observed_data",
    ]

    def test_init(self, sleepstudy):
        sleep_model = Lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)

        # We always make a copy of the data
        assert isinstance(sleep_model.data, pd.DataFrame) and sleep_model.data.equals(
            sleepstudy
        )
        # And store the Bambi model instance
        assert sleep_model.model_obj is not None

        # As well as an arviz inference object (eventually after fitting)
        assert sleep_model.inference_obj is None

        # Model terms are stored as a dictionary
        assert sleep_model.terms == {
            "common_terms": ["Intercept", "Days"],
            "group_terms": ["1|Subject", "Days|Subject"],
            "response_term": "Reaction",
        }

        # Design matrix is a dataframe
        assert isinstance(sleep_model.design_matrix, pd.DataFrame)
        assert sleep_model.grps == {"1|Subject": 18, "Days|Subject": 18}

        # Summary stats for priors over model coefficients are a dataframe organized
        # just like the output of the model.fit() or model.summary()
        assert isinstance(sleep_model.coef_prior, pd.DataFrame)
        assert (
            sleep_model.coef_prior.columns.to_list()
            == self.EXPECTED_PRIOR_MEAN_SUMMARY_COLS
        )

        # We can choose a different summary statistic to summarize priors at
        # initialization time. True for summarizing posteriors as well. Only 'mean' and 'median' are supported.
        sleep_model = Lmer(
            "Reaction ~ Days + (Days | Subject)",
            data=sleepstudy,
            summarize_prior_with="median",
        )
        assert (
            sleep_model.coef_prior.columns.to_list()
            == self.EXPECTED_PRIOR_MEDIAN_SUMMARY_COLS
        )

    def test_fit(self, sleepstudy):
        sleep_model = Lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)

        # By default we print an output summary and return the summarized population
        # parameter posteriors like summary in R()
        out = sleep_model.fit()
        assert isinstance(out, pd.DataFrame)
        assert out.equals(sleep_model.coef_posterior.round(3))

        # We store model diagnostics as a dataframe
        assert isinstance(sleep_model.diagnostics, dict)

        # Fixed effects diagnostics
        assert (
            sleep_model.diagnostics["common_terms"].shape[0] == sleep_model.coef.shape[0]
        )
        # Random effects diagnostics; remember we have 2 random effects terms (intercept and slope)
        assert (
            sleep_model.diagnostics["group_terms"].shape[0]
            == sleep_model.fixef["Subject"].shape[0] * 2
        )

        # For convenience we store: priors, prior predictive samples, posteriors, posterior predictive samples, observed_data, and sample_stats from the inference routine in the same arviz InferenceData object
        assert isinstance(sleep_model.inference_obj, az.InferenceData)
        assert (
            sleep_model.inference_obj.groups()
            == self.EXPECTED_ARVIZ_INFERENCE_DATA_GROUPS
        )

        # Like lme4 we have easy-to-access attributes for fixed and random effects
        # Multiple aliases to get access to a dataframe of summary stats for population
        # level posterior, i.e. the output of .fit() or summary() in R
        assert sleep_model.coef is sleep_model.coef_posterior is sleep_model.coefs
        assert (
            sleep_model.coef.columns.to_list()
            == self.EXPECTED_POSTERIOR_MEAN_SUMMARY_COLS
        )
        assert sleep_model.coef.shape[0] == 2

        # Random effect are deviances from population level parameters
        # These are stored as a dictionary of dataframes with keys corresponding to
        # the random effect terms
        assert isinstance(sleep_model.ranef, dict)
        assert list(sleep_model.ranef.keys()) == ["1|Subject", "Days|Subject"]

        for k in sleep_model.ranef.keys():
            # Each dataframe has as many rows as the number of clusters for that rfx term
            assert sleep_model.ranef[k].shape[0] == sleep_model.grps[k]
            # And the same columns as the population level parameter summary, less p-values
            assert (
                sleep_model.ranef[k].columns.to_list()
                == self.EXPECTED_POSTERIOR_MEAN_RFX_SUMMARY_COLS
            )

        # Fixed effect are cluster level estimates generated by adding the population
        # level estimate to the random effect for each cluster.
        # Where as .ranef estimate are organized by model terms, .fixef are organized
        # by cluster variable name, i.e. '1|Subject' and 'Days|Subject' vs 'Subject'
        # Depending on the complexity of the model formula these can be hard to
        # automatically generate due to column names and data structures.
        # Currently this works for 1 clustering variable
        assert isinstance(sleep_model.fixef, dict)
        assert list(sleep_model.fixef.keys()) == ["Subject"]
        assert isinstance(sleep_model.fixef["Subject"], pd.DataFrame)

        # The number of rows in the fixef dataframe should be the same as the number of
        # rows in the ranef dataframes for the same cluster variable
        assert (
            sleep_model.fixef["Subject"].shape[0]
            == sleep_model.grps["1|Subject"]
            == sleep_model.grps["Days|Subject"]
        )

        # Columns are organized by model terms and include summaries of posterior
        # estimates and uncertainties, e.g. b1_estimate, b1_2.5_hdi, b1_97.5_hdi, b2_estimate, ... etc
        EXPECTED_FIXEF_COLNAMES = []
        for term in sleep_model.terms["common_terms"]:
            EXPECTED_FIXEF_COLNAMES.extend(
                [f"{term}_Estimate", f"{term}_2.5_hdi", f"{term}_97.5_hdi"]
            )
        assert sleep_model.fixef["Subject"].columns.to_list() == EXPECTED_FIXEF_COLNAMES

        # Because we already have the posteriors we can summarize them using the median
        # without refitting the model
        sleep_model.summary(summarize_posterior_with="median", return_summary=False)
        assert sleep_model.coef.shape[0] == 2

        # Run same tests as above but with median summaries
        assert (
            sleep_model.coef.columns.to_list()
            == self.EXPECTED_POSTERIOR_MEDIAN_SUMMARY_COLS
        )
        for k in sleep_model.ranef.keys():
            # Each dataframe has as many rows as the number of clusters for that rfx term
            assert sleep_model.ranef[k].shape[0] == sleep_model.grps[k]
            # And the same columns as the population level parameter summary
            assert (
                sleep_model.ranef[k].columns.to_list()
                == self.EXPECTED_POSTERIOR_MEDIAN_SUMMARY_COLS
            )
        EXPECTED_FIXEF_COLNAMES = []
        for term in sleep_model.terms["common_terms"]:
            EXPECTED_FIXEF_COLNAMES.extend(
                [f"{term}_Estimate", f"{term}_2.5_eti", f"{term}_97.5_eti"]
            )
        assert sleep_model.fixef["Subject"].columns.to_list() == EXPECTED_FIXEF_COLNAMES

        # Finally lets test tweaking how inference is done
        # Default is to use 4 chains and 1000 draws per chain
        assert sleep_model.inference_obj.posterior.sizes == {
            "chain": self.MAX_CHAINS,
            "draw": 1000,
            "Subject__factor_dim": 18,
            "__obs__": 180,
        }

        # We can change this by passing chains and draws to .fit()
        # We can also change other parameters supported by bambi like tune
        # Change num changes and draws
        sleep_model.fit(chains=2, draws=500, summary=False)
        assert sleep_model.inference_obj.posterior.sizes == {
            "chain": 2,
            "draw": 500,
            "Subject__factor_dim": 18,
            "__obs__": 180,
        }

        # Change inference method to pymc's mcmc sampler, which is a bit slower
        # Reduce chains and draws just for testing speed
        sleep_model.fit(chains=2, draws=100, inference_method="mcmc", summary=False)
        assert sleep_model.inference_obj.sample_stats.attrs["inference_library"] == "pymc"

    def test_model_comparison(self, sleepstudy):
        sleep_model = Lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
        sleep_model.fit(perform_model_comparison=True)
        assert isinstance(sleep_model.nested_model_comparison, pd.DataFrame)
        assert (
            sleep_model.nested_model_comparison.shape[0] == sleep_model.coef.shape[0] - 1
        )
        assert (
            sleep_model.nested_model_comparison.columns.to_list()
            == self.EXPECTED_MODEL_COMPARISON_COLS
        )

        # Now just testing fitting and refitting with model comparison to make sure
        # attribute saving is working
        sleep_model = Lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
        sleep_model.fit(summary=False)
        coef_shape = sleep_model.coef_posterior.shape[1]

        # With model comparison we augment model.coefs
        sleep_model.fit(perform_model_comparison=True)
        assert isinstance(sleep_model.nested_model_comparison, pd.DataFrame)
        assert sleep_model.coefs.shape[1] == coef_shape + 2

    def test_sleepstudy_precision(self, sleepstudy):
        """Check numpyro fit against pymc"""

        model = Lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
        model.fit(summary=False)

        # Test against pymc values from bambi tutorial:
        # https://bambinos.github.io/bambi/notebooks/sleepstudy.html#analyze-results
        assert model.coef.loc["Intercept", "2.5_hdi"] > 233
        assert model.coef.loc["Intercept", "97.5_hdi"] < 268
        assert model.coef.loc["Days", "2.5_hdi"] > 6.5
        assert model.coef.loc["Days", "97.5_hdi"] < 15

    def test_sampledata_precision(self, sampledata):
        """Check numpyro fit against lme4"""

        # Test against lme4 model from pymer4 tutorial 1
        model = Lmer("DV ~ IV2 + (IV2|Group)", data=sampledata)
        model.fit(summary=False)

        assert model.coef.loc["Intercept", "2.5_hdi"] > 4
        assert model.coef.loc["Intercept", "97.5_hdi"] < 16
        assert model.coef.loc["IV2", "2.5_hdi"] > 0.5
        assert model.coef.loc["IV2", "97.5_hdi"] < 0.9
