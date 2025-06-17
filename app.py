import streamlit as st
import numpy as np
import pandas as pd
from AutoVisualizer.processing import check_dataset_cleanliness, task_type,is_probably_categorical, is_discrete, is_continuous, parse_datetime_columns
from AutoVisualizer.categorical_viz import combine_figures_as_subplots, generate_count_plots, generate_bar_plots, generate_grouped_bar_plots, generate_pie_plots, generate_categorical_correlation_heatmap
from AutoVisualizer.numerical_viz import generate_box_plots, generate_numeric_correlation_heatmap, generate_scatter_plots, generate_histograms, generate_line_plots

st.set_page_config(page_title= "Auto-Visualizer",page_icon= "📊",layout="wide")
# col1, col2 = st.columns([0.3, 0.7])

with st.sidebar:
    # Upload the dataset file (can upload only CSV, XLSX, JSON, XML)
    uploaded_file = st.file_uploader("Upload your dataset file:", ["csv", "xlsx", "json", "xml"])

if uploaded_file is not None:
    file_type = uploaded_file.name

    try:
        # Read the dataset through pandas
        if file_type.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_type.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif file_type.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            df = pd.read_xml(uploaded_file)
    except Exception as e:
        st.write("Error:", e)
    
    with st.sidebar:
        # Show a a Disclaimer to user to upload as clean data as possible
        st.info("""
            ⚠️ **Heads up!** For the best experience, please upload a clean dataset.
            
            This app is designed for *visualizing data*, not cleaning it.
            
            📌 *Tip:* Use the quick checker below to spot potential issues.
            """)

        if st.button("Run Cleanliness Check"):
            # Use session state to track button click
            st.session_state.run_clean_check = True
        st.divider()

    # Display the cleanliness checker result on the main page (not sidebar) if button was clicked
    if st.session_state.get("run_clean_check", False):
        with st.expander("➡️ See Cleanliness Checker Result"):
            check_dataset_cleanliness(df)

    # Display the user DataFrame
    st.markdown("Your Dataset:")
    st.dataframe(df, height=210)
    st.divider()

    # Extract column names from the dataset + add "No Target" element if there's no target in the user dataset
    feature_list = list(df.columns)
    target_selector = ["No Target"] + feature_list

    with st.sidebar:
        # Ask the user to select target column from their dataset
        target_col = st.selectbox("Specify the target column in your dataset:", target_selector)
        # Identify the task/type of dataset (i.e. classification/regression/clustering(if no target feature at all))
        task = task_type(df, target_col)
        # Display the task to user
        st.write(f"🔍 Task identified: **{task}**")
        
    # Identify the date-time columns (if any) and extract new time-based components from it
    # date_time_ls       --> List that will store date-time feature names
    # extracted_datetime --> List that will store extracted date-time feature names
    df, date_time_ls, extracted_datetime = parse_datetime_columns(df)

    # Remove date-time feature names as we already extracted time based components from it
    feature_list = [x for x in feature_list if x not in date_time_ls]
    
    categorical_ls = []              # List that will store categorical feature names
    discrete_ls = []                 # List that will store discrete feature names
    continuous_ls = []               # List that will store continuous feature names
    for feature in feature_list:
        if is_probably_categorical(df[feature]):
            categorical_ls.append(feature)         # Calling Categorical Feature Identifier Function
        elif is_discrete(df[feature]):
            discrete_ls.append(feature)            # Calling Discrete Feature Identifier Function
        elif is_continuous(df[feature]):
            continuous_ls.append(feature)          # Calling Continuous Feature Identifier Function
    # Add time based components that appear to be categorical
    for feature in extracted_datetime:
        if is_probably_categorical(df[feature]):
            categorical_ls.append(feature)
    
    # Creating Dialog Box to show identified features
    @st.dialog("Identified/Extracted Features from your Dataset:-")
    def open_dialog():
        if categorical_ls:
            with st.popover("Categorical Features", use_container_width= True):
                st.code("\n".join([f"• {item}" for item in categorical_ls]))
        if discrete_ls:
            with st.popover("Discrete Features", use_container_width= True):
                st.code("\n".join([f"• {item}" for item in discrete_ls]))
        if continuous_ls:
            with st.popover("Continuous Features", use_container_width= True):
                st.code("\n".join([f"• {item}" for item in continuous_ls]))
        if date_time_ls:
            with st.popover("Date-Time Features", use_container_width= True):
                st.code("\n".join([f"• {item}" for item in date_time_ls]))
            with st.popover("Extracted features from your Date-Time like features", use_container_width= True):
                st.code("\n".join([f"• {item}" for item in extracted_datetime]))
    with st.sidebar:
        # Calling the dialog box through a button
        if st.button("See Your Feature Details"):
            open_dialog()

    # Generate the Plots
    with st.spinner("Generating Plots.....", show_time= True):
        if categorical_ls:
            st.header("📊 Categorical Plots")
            # 1. Count Plots
            count_plots = []
            for x_col in categorical_ls:
                if df[x_col].nunique() <= 20:
                    count_plots.extend(generate_count_plots(df, x_col))
            
            if count_plots:
                st.subheader("Count Plots :-")
                st.plotly_chart(combine_figures_as_subplots(count_plots), use_container_width=True)
            
            # 2. Bar Plots
            bar_plots = []
            # (Categorical vs Discrete + Continuous)
            for x_col in categorical_ls:
                if df[x_col].nunique() <= 20:
                    bar_plots.extend(generate_bar_plots(df, x_col, discrete_ls + continuous_ls))

            if bar_plots:
                st.subheader("Bar Plots :-")
                st.plotly_chart(combine_figures_as_subplots(bar_plots), use_container_width=True)
            
            # 3. Grouped Bar Plots
            grp_bar_plots = []
            # (Categorical vs Discrete + Continuous)
            grp_bar_plots.extend(generate_grouped_bar_plots(df, categorical_ls, discrete_ls + continuous_ls))

            if grp_bar_plots:
                st.subheader("Grouped Bar Plots :-")
                st.plotly_chart(combine_figures_as_subplots(grp_bar_plots), use_container_width=True)
            
            # 4. Pie Charts
            pie_plots = []
            for x_col in categorical_ls:
                if df[x_col].nunique() <= 20:
                    pie_plots.extend(generate_pie_plots(df, x_col))

            if pie_plots:
                st.subheader("Pie Charts :-")
                st.plotly_chart(combine_figures_as_subplots(pie_plots), use_container_width=True)

        if continuous_ls:
            st.header("📊 Numerical Plots")
            # 5. Box Plots
            box_plots = []
            for x_col in categorical_ls:
                if df[x_col].nunique() <= 10:
                    box_plots.extend(generate_box_plots(df, x_col, continuous_ls))

            if box_plots:
                st.subheader("Box Plots :-")
                st.plotly_chart(combine_figures_as_subplots(box_plots), use_container_width=True)
            
            # 6. Heat Maps
            heat_maps = []
            if task == 'Regression':
                if categorical_ls:
                    heat_maps.extend(generate_categorical_correlation_heatmap(df, target_col, categorical_ls))
            heat_maps.extend(generate_numeric_correlation_heatmap(df[continuous_ls]))
            if heat_maps: 
                st.subheader("Heat Maps :-")
                st.plotly_chart(combine_figures_as_subplots(heat_maps), use_container_width=True)
            
            # 7. Scatter Plots
            if len(continuous_ls) >= 2:
                st.subheader("Scatter Plots")
            scatter_plots = []
            # Creation of unique feature pairs (no repetition like (B, A) if (A, B) is already used)
            feature_pairs = []
            for i in range(len(continuous_ls)):
                for j in range(i + 1, len(continuous_ls)):
                    feature_pairs.append((continuous_ls[i], continuous_ls[j]))
            selection = st.pills("Highlight using a categorical feature :- ", categorical_ls)
            scatter_plots.extend(generate_scatter_plots(df, feature_pairs, selection))
            if scatter_plots:
                st.plotly_chart(combine_figures_as_subplots(scatter_plots), use_container_width=True)
            
            # 8. Histograms
            histograms = []
            histograms.extend(generate_histograms(df, continuous_ls))

            if histograms:
                st.subheader("Histograms")
                st.plotly_chart(combine_figures_as_subplots(histograms), use_container_width=True)
            
            # 9. Line Plots
            line_plots = []
            if date_time_ls:
                # Extract only date-related components
                date_related_keywords = ['_year', '_month', '_day', '_weekday']
                date_component_cols = [col for col in extracted_datetime if any(key in col for key in date_related_keywords)]
                if date_component_cols:
                    st.subheader("Line Plots :-")
                    # Mapping of labels to values
                    time_grouping_options = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "ME",
                        "Yearly": "YE"
                    }
                    time_choice  = st.pills("Choose time interval for grouping :- ", list(time_grouping_options.keys()))
                    # Extract the actual value for resampling
                    selected_freq = time_grouping_options[time_choice] if time_choice else "ME"  # Use default "ME" if no selection
                    line_plots.extend(generate_line_plots(df, date_component_cols, continuous_ls, selected_freq))
            
            if line_plots:
                st.plotly_chart(combine_figures_as_subplots(line_plots), use_container_width= True)
