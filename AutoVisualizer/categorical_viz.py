# Module that will handle plots like bar chart, count plot, pie chart, etc.
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Subplotting Function
def combine_figures_as_subplots(figures: list, rows_per_column: int = 2):
    """
    Combines a list of Plotly figures into a single subplot layout.

    Args:
        figures (list): A list of individual Plotly figure objects to be arranged as subplots.
        rows_per_column (int, optional): The number of subplot rows per column. Defaults to 2.

    Returns:
        plotly.graph_objects.Figure: A single Plotly figure containing all input figures as subplots.
    """
    total = len(figures)
    cols = 1 if total <= rows_per_column else 2
    rows = (total + cols - 1) // cols           # Ceiling division to compute total rows

    # Determine subplot types for each cell
    specs = []
    for i in range(rows):
        row_specs = []
        for j in range(cols):
            idx = i * cols + j
            if idx < total and figures[idx].data and figures[idx].data[0].type == "pie":
                row_specs.append({"type": "domain"})
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)

    # Create subplots with correct specs
    subplot_fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[fig.layout.title.text for fig in figures],
        specs=specs
    )

    # Add each figure's traces to the corresponding subplot cell
    for idx, fig in enumerate(figures):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        for trace in fig.data:
            subplot_fig.add_trace(trace, row=row, col=col)

        # Only update axes for non-pie charts
        if fig.data[0].type != "pie":
            xaxis_title = fig.layout.xaxis.title.text
            yaxis_title = fig.layout.yaxis.title.text
            subplot_fig.update_xaxes(title_text=xaxis_title, row=row, col=col)
            subplot_fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
    
    # Final layout adjustments
    subplot_fig.update_layout(height=350 * rows, showlegend=False)
    return subplot_fig

# Count Plots Function
def generate_count_plots(df: pd.DataFrame, categorical_column: str, max_label_len: int = 10):
    """
    Generates a count plot (bar chart) for a given categorical column in a DataFrame.

    Args:
        df (pd.DataFrame): The input dataset.
        categorical_column (str): The name of the categorical column to plot.
        max_label_len (int, optional): Maximum number of characters to display on x-axis labels. 
                                       Longer labels are truncated with an ellipsis. Defaults to 10.

    Returns:
        list: A list containing a single Plotly bar chart figure.
    """
    plots = []
    # Compute value counts for the selected column
    value_counts = df[categorical_column].value_counts().reset_index()
    value_counts.columns = [categorical_column, "Count"]

    # Add truncated label column
    value_counts["Truncated"] = value_counts[categorical_column].apply(
        lambda x: x if len(str(x)) <= max_label_len else str(x)[:max_label_len] + "…"
    )

    # Create a bar plot
    fig = px.bar(
        value_counts, x="Truncated", y="Count", color="Truncated",
        title=f"Count Plot for '{categorical_column}'",
        labels={categorical_column: "Truncated", "Count": "Frequency"},
        custom_data=[value_counts[categorical_column]]
    )

    # Use full label in hover template
    fig.update_traces(
        hovertemplate=f"{categorical_column}=%{{customdata}}<br>Count=%{{y}}<extra></extra>"
    )

    fig.update_layout(showlegend=False, xaxis_title=categorical_column)
    plots.append(fig)
    return plots

# Bar Plot Function
def generate_bar_plots(df: pd.DataFrame, x_col: str, y_columns: list, max_label_len: int = 10):
    """
    Generates bar plots showing the average of continuous features grouped by a categorical column.

    Args:
        df (pd.DataFrame): The input dataset.
        x_col (str): The categorical column to group by (x-axis).
        y_columns (list): List of continuous columns to aggregate and plot (y-axis).
        max_label_len (int, optional): Maximum number of characters for x-axis labels. 
                                       Labels longer than this are truncated. Defaults to 10.

    Returns:
        list: A list of Plotly bar chart figures.
    """
    plots = []

    for y_col in y_columns:
        # Compute mean aggregation
        agg_func = df.groupby(x_col)[y_col].mean().reset_index()

        # Add truncated label for x-axis ticks
        agg_func["Truncated"] = agg_func[x_col].apply(
            lambda x: x if len(str(x)) <= max_label_len else str(x)[:max_label_len] + "…"
        )

        fig = px.bar(
            agg_func, x="Truncated", y=y_col, color="Truncated",  # color by original to retain legend info
            title=f"{x_col} vs {y_col} (Average)",
            labels={"Truncated": "", y_col: y_col},
            custom_data=[agg_func[x_col]],
        )

        # Format hover values
        if (df[y_col].dropna() % 1 == 0).all():
            hover_format = ".0f"
        else:
            hover_format = ".2f"

        # Show full x value on hover instead of truncated
        fig.update_traces(
            hovertemplate=f"{x_col}=%{{customdata[0]}}<br>Average {y_col}=%{{y:{hover_format}}}<extra></extra>",
        )

        fig.update_layout(showlegend=False, xaxis_title=x_col)
        plots.append(fig)

    return plots

# Grouped Bar Plot Function
def generate_grouped_bar_plots(df: pd.DataFrame, x_columns: list, y_columns: list, max_label_len: int = 10):
    """
    Generates grouped bar plots showing the average of continuous features 
    grouped by a categorical feature and further separated by a binary hue column.

    Args:
        df (pd.DataFrame): The input dataset.
        x_columns (list): Categorical columns to consider for x-axis and hue roles.
        y_columns (list): Continuous columns to be averaged and plotted.
        max_label_len (int, optional): Maximum length of x-axis labels before truncation. Defaults to 10.

    Returns:
        list: A list of Plotly grouped bar chart figures.
    """
    plots = []
    
    # Split categorical columns
    hue_candidates = [col for col in x_columns if df[col].nunique() <= 2]
    x_candidates = [col for col in x_columns if df[col].nunique() > 2 and df[col].nunique() <= 10]
    
    for x_col in x_candidates:
        for y_col in y_columns:
            for hue_col in hue_candidates:
                # Group and aggregate
                agg_func = df.groupby([x_col, hue_col])[y_col].mean().reset_index()

                # Add truncated label for x-axis ticks
                agg_func["Truncated"] = agg_func[x_col].apply(
                    lambda x: x if len(str(x)) <= max_label_len else str(x)[:max_label_len] + "…"
                )

                fig = px.bar(
                    agg_func,
                    x="Truncated",
                    y=y_col,
                    color=hue_col,
                    barmode="group",
                    title=f"{x_col} vs {y_col} grouped by {hue_col}",
                    labels={x_col: x_col, y_col: f"Average {y_col}", hue_col: hue_col},
                    custom_data=[agg_func[x_col], agg_func[hue_col]]
                )

                # Formatting hover text
                if (df[y_col].dropna() % 1 == 0).all():
                    hover_format = ".0f"
                else:
                    hover_format = ".2f"
            
                # Show full x value on hover instead of truncated
                fig.update_traces(
                    hovertemplate=f"{x_col}=%{{customdata[0]}}<br>{hue_col}=%{{customdata[1]}}<br>Average {y_col}=%{{y:{hover_format}}}<extra></extra>",
                )
                plots.append(fig)

    return plots

# Pie Charts Function
def generate_pie_plots(df: pd.DataFrame, categorical_column: str, max_label_len: int = 10):
    """
    Generates a pie chart showing the distribution of values for a given categorical column.

    Args:
        df (pd.DataFrame): The input dataset.
        categorical_column (str): The categorical column for which to create a pie chart.
        max_label_len (int, optional): Maximum length of labels (currently unused here but reserved for consistency). Defaults to 10.

    Returns:
        list: A list containing one Plotly pie chart figure.
    """
    plots = []
    # Count occurrences of each category
    value_counts = df[categorical_column].value_counts().reset_index()
    value_counts.columns = [categorical_column, "Count"]

    fig = px.pie(
        value_counts, names=categorical_column, values="Count",
        hole=0.3, 
        title=f"Pie Plot for '{categorical_column}'",
    )
    # Use full label in hover template
    fig.update_traces(
        textinfo='value',
        hovertemplate=f"{categorical_column}=%{{label}}<br>Percentage=%{{percent}}<extra></extra>"
    )

    fig.update_layout(showlegend=False)
    plots.append(fig)
    return plots

# HeatMap Function
# Heatmap showing correlated categorical features based on a numerical target column
def generate_categorical_correlation_heatmap(df: pd.DataFrame, numerical_col: str, categorical_columns: list):
    """
    Generates a heatmap showing correlations between categorical features based on a numeric target column.

    This function encodes each categorical feature by replacing its categories with the mean value
    of the numeric column for that category. Then it computes the Pearson correlation between the
    encoded categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_col (str): The target numeric column to base encoding on.
        categorical_columns (list): List of categorical feature names to analyze.

    Returns:
        list: A list containing one Plotly heatmap figure visualizing the correlation matrix.
    """
    plots = []
    encoded_df = pd.DataFrame()

    # Encode categorical features using mean of numeric target per category
    for col in categorical_columns:
        temp = df[[col, numerical_col]].dropna()
        means = temp.groupby(col)[numerical_col].mean()
        encoded_df[col] = temp[col].map(means)
    
    # Compute correlation matrix on the encoded values
    corr_matrix = encoded_df.corr(method="pearson")

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        title=f"Categorical Feature Correlation Heatmap Based on your Target : '{numerical_col}'",
    )

    plots.append(fig)
    return plots
