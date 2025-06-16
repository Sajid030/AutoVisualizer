# Module that will handle plots like histogram, boxplot, density plot, etc.
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import gaussian_kde

# Box Plot Function
def generate_box_plots(df: pd.DataFrame, x_col: str, y_columns: list, max_label_len: int = 10):
    plots = []

    df = df[[x_col] + y_columns].dropna()
    
    # Truncate long category names
    df["Truncated_X"] = df[x_col].apply(
        lambda x: str(x) if len(str(x)) <= max_label_len else str(x)[:max_label_len] + "…"
    )

    for y_col in y_columns:
        fig = px.box(
            df,
            x="Truncated_X", y=y_col,
            color="Truncated_X",
            points="outliers",  # Show outliers
            title=f"Box Plot of {y_col} by {x_col}",
            labels={"Truncated_X": x_col, y_col: y_col},
        )

        fig.update_layout(showlegend=False)
        plots.append(fig)

    return plots

# HeatMap Functions
## Heatmap of numerical (continuous) features
def generate_numeric_correlation_heatmap(df: pd.DataFrame):
    plots = []
    
    # Filter numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return plots  # Not enough numeric features to compute correlation

    corr_matrix = numeric_df.corr(method="pearson")

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Numerical Feature Correlation Heatmap",
    )

    plots.append(fig)
    return plots

# Scatter Plot Function
def generate_scatter_plots(df: pd.DataFrame, feature_pairs: list, color_by: str = None):
    plots = []

    for x_col, y_col in feature_pairs:
        fig = px.scatter(
            df.dropna(subset=[x_col, y_col]),
            x=x_col,
            y=y_col,
            color=color_by if color_by in df.columns else None,
            title=f"Scatter Plot: {x_col} vs {y_col}",
            labels={x_col: x_col, y_col: y_col}
        )
        if color_by:
            fig.update_layout(title = f"Scatter Plot: {x_col} vs {y_col} color by {color_by}",showlegend=bool(color_by))
        
        fig.update_layout(showlegend=bool(color_by))
        plots.append(fig)

    return plots

# Histogram Function
def generate_histograms(df: pd.DataFrame, numeric_columns: list, bins: int = 30):
    plots = []

    for col in numeric_columns:
        data = df[col].dropna()

        # Histogram
        hist = go.Histogram(
            x=data,
            nbinsx=bins,
            name='Histogram',
            opacity=0.6
        )

        # KDE Curve
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_curve = go.Scatter(
            x=x_range,
            y=kde(x_range) * len(data) * (data.max() - data.min()) / bins,  # scaled to match histogram
            name='KDE',
            mode='lines',
            line=dict(color='red')
        )

        fig = go.Figure(data=[hist, kde_curve])
        fig.update_layout(
            title=f"Histogram + KDE for '{col}'",
            xaxis_title=col,
            yaxis_title="Count",
            barmode='overlay',
            showlegend=True
        )

        plots.append(fig)

    return plots

# Line Plots Function
def generate_line_plots(df: pd.DataFrame, date_component_cols: list, y_columns: list, freq: str = "M"):
    """
    Generates line plots for each group of datetime component columns.
    Groups like 'order_date_year', 'order_date_month', etc. are reconstructed into a datetime.
    
    freq options:
    - 'D' : daily
    - 'W' : weekly
    - 'M' : monthly
    - 'Y' : yearly
    """
    from collections import defaultdict
    plots = []

    # Group columns by prefix
    groups = defaultdict(dict)
    for col in date_component_cols:
        for suffix in ['_year', '_month', '_day', '_weekday']:
            if col.endswith(suffix):
                prefix = col.replace(suffix, '')
                groups[prefix][suffix] = col

    # Reconstruct datetime and generate plots
    for prefix, components in groups.items():
        if all(k in components for k in ['_year', '_month', '_day']):
            # Build a datetime column from year, month, day
            temp_df = df[[components['_year'], components['_month'], components['_day']] + y_columns].dropna()
            temp_df["__date__"] = pd.to_datetime({
                'year': temp_df[components['_year']],
                'month': temp_df[components['_month']],
                'day': temp_df[components['_day']]
            }, errors='coerce')
            temp_df = temp_df.dropna(subset=["__date__"])
            temp_df.set_index("__date__", inplace=True)

            # Resample and plot
            resampled_df = temp_df.resample(freq).mean().reset_index()

            for col in y_columns:
                fig = px.line(
                    resampled_df,
                    x="__date__",
                    y=col,
                    title=f"Line Plot: {prefix} vs {col}",
                    labels={"__date__": "Date", col: col},
                )
                plots.append(fig)
    
    return plots
