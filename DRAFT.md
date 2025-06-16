# 📝 Project Draft: Dataset Visualizer

## 💡 Project Idea
Build a **Streamlit-based web app** that helps users visualize and understand their datasets. The tool enables users to:
1. Upload a dataset (CSV format).
2. Select target column (or none if unsupervised) to identify problem type.
3. User Dataset columns identification.
4. Automatically view EDA (Exploratory Data Analysis) visualizations.
5. Explore dataset with their own questions using a custom plotting interface.

This will help users explore relationships, distributions, and key characteristics of their data before they start modeling.

---

## ✅ Key Features

### 1. **Upload Dataset**
- User uploads CSV file.
- The app provides the user with a option to run basic cleanliness checks — identifying missing values, duplicate rows, constant columns, and columns with only nulls, etc.

### 2. **Target Column Selection**
- After uploading, the app extracts all column names.
- User selects one as the **target column**, or selects **No Target** if there is no target.
- Based on target column data type:
  - **Numeric:** Regression
  - **Categorical:** Classification
  - **"No Target" selected:** Clustering

### 3. **Features Identification / Column Categorization**
- Automatically identify the type of each column to streamline visualization and analysis.
- Columns are analyzed as:
  - **Numerical**
  - **Categorical**
    - A column is considered **categorical** only if:
      - `nunique <= 50` AND
      - `nunique / total rows <= 0.15`
  - **Non-categorical string features** (like date/time, names, IDs, etc.)

### 4. **Automated Visualizations**
- App generates:
  - Countplots
  - Barplots
  - Grouped Barplots
  - Pie Charts
  - Boxplots
  - Correlation heatmaps
  - Scatterplots
  - Histograms
  - Line plots
- These are useful for gain insights, identifying patterns and outliers, etc.

### 5. **Custom Visualization**
- User selects:
  - Plot type (scatter, bar, line, heatmap, etc.)
  - Features for X, Y, hue (if needed)
  - Filters, bins, and grouping options (optional)
- App renders the plot.
- This allows the user to explore based on what's in their mind.

---

## 🔧 Tech Stack
- **Frontend/UI**: Streamlit
- **Backend Logic**: Python (Pandas, Plotly)
- **Structure**:
  - `app.py`: UI logic
  - `AutoVisualizer/`: Helper modules (data processing, auto viz, custom viz)

<!-- ---

## 🚧 Future Improvements
- Add support for Excel/JSON files
- Feature selection suggestions (based on correlation)
- Auto-detect and visualize time-series trends
- Export analysis reports (PDF/HTML)

--- -->
