import numpy as np
from great_tables import GT, md, loc, style
import polars.selectors as cs
from polars import col
import polars as pl

__all__ = [
    "summary_lm_table",
    "summary_glm_table",
    "summary_lmm_table",
    "summary_glmm_table",
    "anova_table",
    "compare_anova_table",
]


def _sig_stars(val):
    """Adds sig stars to coef table prettier output."""
    star = ""
    if 0 <= val < 0.001:
        star = "***"
    elif 0.001 <= val < 0.01:
        star = "**"
    elif 0.01 <= val < 0.05:
        star = "*"
    elif 0.05 <= val < 0.1:
        star = "."
    return star


def summary_lm_table(model, decimals=2):
    """
    Create a summary table for a model.
    """

    df_model = model.result_fit_stats["df"].item()
    df_model = int(df_model) if df_model else "null"

    nobs = model.result_fit_stats["nobs"].item()
    nobs = int(nobs) if nobs else "null"

    df_resid = model.result_fit_stats["df_residual"].item()
    df_resid = int(df_resid) if df_resid else "null"

    fstat = model.result_fit_stats["statistic"].item()
    fstat = np.round(fstat, decimals) if fstat else "null"

    fstat_p = model.result_fit_stats["p_value"].item()
    fstat_p = np.round(fstat_p, decimals + 1) if fstat_p else "null"
    fstat_p = "<.001" if not isinstance(fstat_p, str) and fstat_p < 0.001 else fstat_p

    resid_error = model.result_fit_stats["sigma"].item()
    resid_error = np.round(resid_error, decimals) if resid_error else "null"

    r_squared = model.result_fit_stats["r_squared"].item()
    r_squared = np.round(r_squared, decimals + 1) if r_squared else "null"

    adj_r_squared = model.result_fit_stats["adj_r_squared"].item()
    adj_r_squared = np.round(adj_r_squared, decimals + 1) if adj_r_squared else "null"

    log_likelihood = model.result_fit_stats["logLik"].item()
    log_likelihood = int(log_likelihood) if log_likelihood else "null"

    aic = model.result_fit_stats["AIC"].item()
    aic = int(aic) if aic else "null"

    bic = model.result_fit_stats["BIC"].item()
    bic = int(bic) if bic else "null"

    table = model.result_fit
    stars = np.array(list(map(_sig_stars, model.result_fit["p_value"].to_numpy())))
    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    non_numeric = ["term", "stars", "p_value"]
    pcol = "p_value"
    table = table.with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
    )

    if model._fit_kwargs["conf_method"] == "boot":
        subtitle = md(
            f"""Number of observations: *{nobs}*  
            Confidence intervals: *{model._fit_kwargs["conf_method"]}*  
            Bootstrap Iterations: *{model._fit_kwargs["nboot"]}*  
            ---------------------  
            R-squared: *{r_squared}*  
            R-squared-adj: *{adj_r_squared}*  
            *F({df_model}, {df_resid}) = {fstat}, p = {fstat_p}*  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )
    else:
        subtitle = md(
            f"""Number of observations: *{nobs}*  
            Confidence intervals: *parametric*  
            ---------------------  
            R-squared: *{r_squared}*  
            R-squared-adj: *{adj_r_squared}*  
            *F({df_model}, {df_resid}) = {fstat}, p = {fstat_p}*  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )

    out = (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(0.75)
        .tab_stub(rowname_col="term")
        .tab_header(title=f"Formula: lm({model.formula})", subtitle=subtitle)
        .fmt_number(
            columns=cs.exclude(non_numeric),
            decimals=decimals,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
        )
        .fmt_number(columns=["df"], decimals=0)
        .tab_style(
            locations=loc.stub(),
            style=style.text(style="italic"),
        )
        .cols_label(
            stars="",
            estimate="Estimate",
            conf_low="CI-low",
            conf_high="CI-high",
            std_error="SE",
            t_stat="T-stat",
            p_value="p",
        )
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
    return out


def summary_glm_table(model, show_odds=False, decimals=2):
    df_null = model.result_fit_stats["df_null"].item()
    df_null = int(df_null) if df_null else "null"

    df_resid = model.result_fit_stats["df_residual"].item()
    df_resid = int(df_resid) if df_resid else "null"

    null_deviance = model.result_fit_stats["null_deviance"].item()
    null_deviance = int(null_deviance) if null_deviance else "null"

    resid_deviance = model.result_fit_stats["deviance"].item()
    resid_deviance = int(resid_deviance) if resid_deviance else "null"

    log_likelihood = model.result_fit_stats["logLik"].item()
    log_likelihood = int(log_likelihood) if log_likelihood else "null"

    aic = model.result_fit_stats["AIC"].item()
    aic = int(aic) if aic else "null"

    bic = model.result_fit_stats["BIC"].item()
    bic = int(bic) if bic else "null"

    if show_odds:
        table = model.result_fit_odds
    else:
        table = model.result_fit
    stars = np.array(list(map(_sig_stars, model.result_fit["p_value"].to_numpy())))
    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    non_numeric = ["term", "stars", "p_value"]
    pcol = "p_value"
    table = table.with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
    )

    if model._fit_kwargs["conf_method"] == "boot":
        subtitle = md(
            f"""Family: *{model.family} (link: *{model.link}*)*  
            Number of observations: *{model.result_fit_stats["nobs"].item()}*  
            Confidence intervals: *{model._fit_kwargs["conf_method"]}*  
            Bootstrap Iterations: *{model._fit_kwargs["nboot"]}*   
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual deviance: *{resid_deviance}*  
        """
        )
    else:
        subtitle = md(
            f"""Family: *{model.family} (link: *{model.link}*)*  
            Number of observations: *{model.result_fit_stats["nobs"].item()}*  
            Confidence intervals: *parametric*  
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual deviance: *{resid_deviance}* | DF: *{df_resid}*  
            Null deviance: *{null_deviance}* | DF: *{df_null}*  
        """
        )

    out = (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(0.75)
        .tab_stub(rowname_col="term")
        .tab_header(title=f"Formula: glm({model.formula})", subtitle=subtitle)
        .fmt_number(
            columns=cs.exclude(non_numeric),
            decimals=decimals,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
        )
        .tab_style(
            locations=loc.stub(),
            style=style.text(style="italic"),
        )
        .cols_label(
            stars="",
            estimate="Estimate",
            conf_low="CI-low",
            conf_high="CI-high",
            std_error="SE",
            z_stat="Z-stat",
            p_value="p",
        )
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
    return out


def summary_lmm_table(model, decimals=2):
    """
    Create a summary table for a model.
    """

    df_resid = model.result_fit_stats["df_residual"].item()
    df_resid = int(df_resid) if df_resid else "null"

    nobs = model.result_fit_stats["nobs"].item()

    log_likelihood = model.result_fit_stats["logLik"].item()
    log_likelihood = int(log_likelihood) if log_likelihood else "null"

    aic = model.result_fit_stats["AIC"].item()
    aic = int(aic) if aic else "null"

    bic = model.result_fit_stats["BIC"].item()
    bic = int(bic) if bic else "null"

    if "REMLcrit" in model.result_fit_stats.columns:
        reml = model.result_fit_stats["REMLcrit"].item()
        reml = np.round(reml, decimals) if reml else "null"
    else:
        reml = "null"

    resid_error = model.result_fit_stats["sigma"].item()
    resid_error = np.round(resid_error, decimals) if resid_error else "null"

    # Fixed effects
    table = model.result_fit
    stars = np.array(list(map(_sig_stars, model.result_fit["p_value"].to_numpy())))
    pcol = "p_value"
    table = table.with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
    )

    # Random effects
    rfx = model.ranef_var.with_columns(
        col("term")
        .str.split("__")
        .list.to_struct(fields=["statistic", "term"])
        .struct.unnest()
    ).with_columns(col("term").str.split(".").list.get(0))

    if model._fit_kwargs["conf_method"] == "satterthwaite":
        rfx = rfx.select("group", "statistic", "term", "estimate")
    else:
        rfx = rfx.select(
            "group", "statistic", "term", "estimate", "conf_low", "conf_high"
        )
    # Sort sds before correlations
    non_resid = rfx.filter(col("group") != "Residual").sort(
        by="statistic", descending=True
    )
    resid = rfx.filter(col("group") == "Residual")
    # Append residuals to end
    rfx = pl.concat([non_resid, resid])
    rfx = rfx.with_columns(
        group=pl.concat_str([col("group"), col("statistic")], separator="-")
    ).drop("statistic")

    # Combine
    empty = pl.DataFrame(np.array([None] * rfx.width)[np.newaxis, :], schema=rfx.schema)
    rfx = pl.concat([rfx, empty, empty], how="vertical")
    rfx[-1, 0] = "Fixed Effects:"
    table = pl.concat([rfx, table], how="diagonal_relaxed")
    table = table.rename({"group": "rfx", "term": "param"})

    if model._fit_kwargs["conf_method"] == "boot":
        subtitle = md(
            f"""Number of observations: *{nobs}*  
            Confidence intervals: *{model._fit_kwargs["conf_method"]}*  
            Bootstrap Iterations: *{model._fit_kwargs["nboot"]}*  
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )
    else:
        subtitle = md(
            f"""Number of observations: *{nobs}*  
            Confidence intervals: *parametric*  
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )

    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    non_numeric = ["rfx", "statistic", "param", "stars", "p_value"]

    out = (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(0.75)
        .tab_header(title=f"Formula: lmer({model.formula})", subtitle=subtitle)
        .fmt_number(
            columns=cs.exclude(non_numeric),
            decimals=decimals,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
        )
        .tab_style(
            locations=loc.body(columns=["param"]),
            style=style.text(style="italic"),
        )
        .cols_label(
            rfx="Random Effects:",
            estimate="Estimate",
            conf_low="CI-low",
            conf_high="CI-high",
            std_error="SE",
            t_stat="T-stat",
            df="DF",
            param="",
            stars="",
            p_value="p",
        )
        .sub_missing(missing_text="")
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
    return out


def summary_glmm_table(model, show_odds=False, decimals=2):
    """
    Create a summary table for a model.
    """

    df_resid = model.result_fit_stats["df_residual"].item()
    df_resid = int(df_resid) if df_resid else "null"

    nobs = model.result_fit_stats["nobs"].item()

    log_likelihood = model.result_fit_stats["logLik"].item()
    log_likelihood = int(log_likelihood) if log_likelihood else "null"

    aic = model.result_fit_stats["AIC"].item()
    aic = int(aic) if aic else "null"

    bic = model.result_fit_stats["BIC"].item()
    bic = int(bic) if bic else "null"

    if "REMLcrit" in model.result_fit_stats.columns:
        reml = model.result_fit_stats["REMLcrit"].item()
        reml = np.round(reml, decimals) if reml else "null"
    else:
        reml = "null"

    resid_error = model.result_fit_stats["sigma"].item()
    resid_error = np.round(resid_error, decimals) if resid_error else "null"

    # Fixed effects
    if show_odds:
        table = model.result_fit_odds
    else:
        table = model.result_fit
    stars = np.array(list(map(_sig_stars, model.result_fit["p_value"].to_numpy())))
    pcol = "p_value"
    table = table.with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
    )

    # Random effects
    rfx = model.ranef_var.with_columns(
        col("term")
        .str.split("__")
        .list.to_struct(fields=["statistic", "term"])
        .struct.unnest()
    ).with_columns(col("term").str.split(".").list.get(0))

    if model._fit_kwargs["conf_method"] == "satterthwaite":
        rfx = rfx.select("group", "statistic", "term", "estimate")
    else:
        rfx = rfx.select(
            "group", "statistic", "term", "estimate", "conf_low", "conf_high"
        )
    # Sort sds before correlations
    non_resid = rfx.filter(col("group") != "Residual").sort(
        by="statistic", descending=True
    )
    resid = rfx.filter(col("group") == "Residual")
    # Append residuals to end
    rfx = pl.concat([non_resid, resid])
    rfx = rfx.with_columns(
        group=pl.concat_str([col("group"), col("statistic")], separator="-")
    ).drop("statistic")

    # Combine
    empty = pl.DataFrame(np.array([None] * rfx.width)[np.newaxis, :], schema=rfx.schema)
    rfx = pl.concat([rfx, empty, empty], how="vertical")
    rfx[-1, 0] = "Fixed Effects:"
    table = pl.concat([rfx, table], how="diagonal_relaxed")
    table = table.rename({"group": "rfx", "term": "param"})

    if model._fit_kwargs["conf_method"] == "boot":
        subtitle = md(
            f"""Family: *{model.family} (link: *{model.link}*)*  
            Number of observations: *{nobs}*  
            Confidence intervals: *{model._fit_kwargs["conf_method"]}*   
            Bootstrap Iterations: *{model._fit_kwargs["nboot"]}*  
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )
    else:
        subtitle = md(
            f"""Family: *{model.family} (link: *{model.link}*)*  
            Number of observations: *{model.result_fit_stats["nobs"].item()}*  
            Confidence intervals: *parametric*  
            ---------------------  
            Log-likelihood: *{log_likelihood}*  
            AIC: *{aic}* | BIC: *{bic}*  
            Residual error: *{resid_error}*  
        """
        )

    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    non_numeric = ["rfx", "statistic", "param", "stars", "p_value"]

    out = (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(0.75)
        .tab_header(title=f"Formula: glmer({model.formula})", subtitle=subtitle)
        .fmt_number(
            columns=cs.exclude(non_numeric),
            decimals=decimals,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
        )
        .tab_style(
            locations=loc.body(columns=["param"]),
            style=style.text(style="italic"),
        )
        .cols_label(
            rfx="Random Effects:",
            estimate="Estimate",
            conf_low="CI-low",
            conf_high="CI-high",
            std_error="SE",
            z_stat="Z-stat",
            param="",
            stars="",
            p_value="p",
        )
        .sub_missing(missing_text="")
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
    return out


def anova_table(model, decimals=2):
    stars = np.array(list(map(_sig_stars, model.result_anova["p_value"].to_numpy())))
    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    pcol = "p_value"
    table = model.result_anova.with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
    )
    table = (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(1)
        .tab_header(title="ANOVA (Type III tests)")
        .fmt_number(
            columns=cs.exclude(["model term", "stars", "p_value"]),
            decimals=decimals,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
        )
        .cols_label(stars="")
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
    if not hasattr(model, "ranef_var"):
        table = table.fmt_integer(columns=cs.starts_with("df"))
    return table


def compare_anova_table(result, *models, decimals=2):
    subtitle = ""
    for i, m in enumerate(models):
        if m.family:
            if hasattr(m, "ranef"):
                subtitle += f"Model {i + 1}: glmer({m.formula})   \n"
            else:
                subtitle += f"Model {i + 1}: glm({m.formula})   \n"
        else:
            if hasattr(m, "ranef"):
                subtitle += f"Model {i + 1}: lmer({m.formula})   \n"
            else:
                subtitle += f"Model {i + 1}: lm({m.formula})   \n"

    subtitle = md(subtitle)
    # NOTE: we use the last column because depending on the test type the name changes
    # PR(>F) or PR(>Chisq)
    stars = np.array(list(map(_sig_stars, result[:, -1].to_numpy())))
    sig_codes = md("Signif. codes: *0 *** 0.001 ** 0.01 * 0.05 . 0.1*")
    ids = np.array(range(len(models))) + 1

    pcol = result.columns[-1]
    table = result.fill_null(0.0).with_columns(
        pl.when(col(pcol).lt(0.001).and_(col(pcol).gt(0.0)))
        .then(pl.lit("<.001"))
        .otherwise(
            pl.when(col(pcol).eq(0.0))
            .then(col(pcol).cast(str).replace_strict(old="0.0", new="", default=""))
            .otherwise(col(pcol).round_sig_figs(decimals + 1).cast(str))
        )
        .alias(pcol),
        stars=stars,
        ids=ids,
    )

    if "npar" in table.columns:
        table = table.with_columns(col("npar").cast(int))

    return (
        GT(table)
        .opt_align_table_header("left")
        .opt_vertical_padding(1)
        .tab_header(title="Analysis of Deviance Table", subtitle=subtitle)
        .fmt_number(
            columns=cs.numeric().exclude(["AIC", "BIC", "logLik"]),
            decimals=decimals,
            use_seps=False,
            drop_trailing_zeros=False,
            drop_trailing_dec_mark=True,
            compact=True,
        )
        .fmt_number(
            columns=["AIC", "BIC", "logLik"],
            decimals=1,
            use_seps=False,
            drop_trailing_zeros=True,
            drop_trailing_dec_mark=True,
        )
        .sub_zero(zero_text="")
        .cols_move_to_start(columns="ids")
        .tab_stub(rowname_col="ids")
        .cols_label(stars="", ids="")
        .tab_source_note(source_note=sig_codes)
        .tab_options(source_notes_font_size="small")
    )
