# Module that will handle plots like histogram, boxplot, density plot, etc.
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import gaussian_kde

# Box Plot Function
def generate_box_plots(df: pd.DataFrame, x_col: str, y_columns: list, max_label_len: int = 10):
    """
    Generates box plots for multiple numerical columns grouped by a categorical column.

    This function creates a series of Plotly box plots to visualize the distribution,
    spread, and potential outliers of each numerical column across the values of a given 
    categorical column. Long category labels are truncated for better readability.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): Categorical column to group the box plots by.
        y_columns (list): List of numerical columns to plot on the y-axis.
        max_label_len (int, optional): Maximum character length for category labels on the x-axis (default is 10).

    Returns:
        list: A list of Plotly figure objects, each representing a box plot for a y_column grouped by x_col.
    """
    plots = []

    # Drop rows with missing values in relevant columns
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
    """
    Generates a heatmap to visualize Pearson correlation between numerical features.

    This function computes the correlation matrix for all numeric columns in the
    DataFrame and returns a heatmap figure representing the strength and direction
    of linear relationships between features.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list containing a single Plotly heatmap figure if enough numeric 
              features exist; otherwise, returns an empty list.
    """
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
    """
    Generates scatter plots for given pairs of numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features.
        feature_pairs (list): List of tuples containing (x, y) column names for each plot.
        color_by (str, optional): Column name to color points by. Must be in df. Defaults to None.

    Returns:
        list: A list of Plotly scatter plot figures, one for each feature pair.
    """
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
        # Update title and legend if coloring is applied
        if color_by:
            fig.update_layout(title = f"Scatter Plot: {x_col} vs {y_col} color by {color_by}",showlegend=bool(color_by))
        
        fig.update_layout(showlegend=bool(color_by))
        plots.append(fig)

    return plots

# Histogram Function
def generate_histograms(df: pd.DataFrame, numeric_columns: list, bins: int = 30):
    """
    Generates histogram and KDE plots for each specified numeric column.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric features.
        numeric_columns (list): List of column names to plot histograms for.
        bins (int, optional): Number of bins to use in the histogram. Defaults to 30.

    Returns:
        list: A list of Plotly figures, each showing a histogram and KDE curve.
    """
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

        # Combine into a single figure
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
    Generates line plots for given numerical columns based on reconstructed datetime columns.

    This function identifies datetime-related components in the DataFrame (e.g., 'order_year', 'order_month', 'order_day'),
    reconstructs them into actual datetime objects, and then plots the specified `y_columns` over time using the chosen frequency.

    Args:
        df (pd.DataFrame): The input DataFrame containing date components and numerical data.
        date_component_cols (list): List of column names representing datetime parts (e.g., 'order_date_year', 'order_date_month').
        y_columns (list): List of numeric columns to be plotted against the date.
        freq (str, optional): Resampling frequency for datetime aggregation.
            Options:
                'D' = daily
                'W' = weekly
                'M' = monthly (default)
                'Y' = yearly

    Returns:
        list: A list of Plotly line plot figures for each (prefix, y_column) pair.
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
