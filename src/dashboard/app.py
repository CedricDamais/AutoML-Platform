import streamlit as st
import requests
import pandas as pd
import time
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001")

st.set_page_config(
    page_title="AutoML Platform Dashboard", page_icon="🚀", layout="wide"
)

st.title("🚀 AutoML Platform Dashboard")


# Helper function for displaying runs
def display_runs(runs, dataset_key="default"):
    """Display runs in a nice formatted way

    Args:
        runs: List of MLflow run objects
        dataset_key: Unique identifier for the dataset (used for widget keys)
    """

    # Prepare data for comparison table
    comparison_data = []

    for run in runs:
        data = run.get("data", {})
        info = run.get("info", {})

        # Extract tags
        tags = {tag["key"]: tag["value"] for tag in data.get("tags", [])}

        # Extract metrics
        metrics = {metric["key"]: metric["value"] for metric in data.get("metrics", [])}

        # Extract params
        params = {param["key"]: param["value"] for param in data.get("params", [])}

        # Shorten run name by removing UUID
        full_run_name = info.get("run_name", "Unnamed")
        short_run_name = (
            full_run_name.split("_")[0] if "_" in full_run_name else full_run_name
        )

        run_info = {
            "Run Name": full_run_name,
            "Display Name": short_run_name,
            "Model": tags.get("model_type", "Unknown"),
            "Framework": tags.get("framework", "Unknown"),
            "Task": tags.get("task_type", "Unknown"),
            "Status": info.get("status", "Unknown"),
        }

        # Add key metrics based on task type
        if tags.get("task_type") == "classification":
            run_info["Test Accuracy"] = metrics.get("test_accuracy", 0)
            run_info["Test F1"] = metrics.get("test_f1", 0)
        else:
            run_info["Test MAE"] = metrics.get("test_mae", 0)
            run_info["Test RMSE"] = metrics.get("test_rmse", 0)
            run_info["Test R²"] = metrics.get("test_r2", 0)

        # Add some params for context
        if "n_estimators" in params:
            run_info["n_estimators"] = params["n_estimators"]
        if "learning_rate" in params:
            run_info["learning_rate"] = params["learning_rate"]

        comparison_data.append(run_info)

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # Highlight best model
        task_type = comparison_data[0].get("Task", "classification")

        # 🎨 ENHANCED VISUAL SECTION
        st.markdown("---")

        # Best model banner with custom styling
        if task_type == "classification" and "Test Accuracy" in df.columns:
            best_idx = df["Test Accuracy"].idxmax()
            best_model = df.loc[best_idx, "Display Name"]
            best_score = df.loc[best_idx, "Test Accuracy"]

            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 25px; border-radius: 15px; text-align: center;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                <h2 style="color: white; margin: 0;">🏆 Best Model</h2>
                <h1 style="color: #ffd700; margin: 10px 0; font-size: 2.5em;">{best_model}</h1>
                <h3 style="color: white; margin: 0;">Accuracy: {best_score:.2%}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif "Test R²" in df.columns:
            best_idx = df["Test R²"].idxmax()
            best_model = df.loc[best_idx, "Display Name"]
            best_score = df.loc[best_idx, "Test R²"]

            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 25px; border-radius: 15px; text-align: center;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                <h2 style="color: white; margin: 0;">🏆 Best Model</h2>
                <h1 style="color: #ffd700; margin: 10px 0; font-size: 2.5em;">{best_model}</h1>
                <h3 style="color: white; margin: 0;">R² Score: {best_score:.4f}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 📊 INTERACTIVE CHARTS SECTION
        st.subheader("📊 Interactive Model Comparison")

        # Create interactive bar chart for metrics
        if task_type == "classification" and "Test Accuracy" in df.columns:
            # Multi-metric comparison for classification
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Test Accuracy by Model", "Test F1 Score by Model"),
                specs=[[{"type": "bar"}, {"type": "bar"}]],
            )

            # Accuracy chart
            fig.add_trace(
                go.Bar(
                    x=df["Display Name"],
                    y=df["Test Accuracy"],
                    name="Accuracy",
                    marker_color="rgb(102, 126, 234)",
                    text=df["Test Accuracy"].apply(lambda x: f"{x:.2%}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2%}<extra></extra>",
                    customdata=df["Run Name"],
                    hovertext=df["Run Name"],
                ),
                row=1,
                col=1,
            )

            # F1 Score chart
            fig.add_trace(
                go.Bar(
                    x=df["Display Name"],
                    y=df["Test F1"],
                    name="F1 Score",
                    marker_color="rgb(118, 75, 162)",
                    text=df["Test F1"].apply(lambda x: f"{x:.2%}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>F1: %{y:.2%}<extra></extra>",
                    customdata=df["Run Name"],
                    hovertext=df["Run Name"],
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Classification Metrics Comparison",
                title_x=0.5,
                hovermode="closest",
            )
            fig.update_xaxes(tickangle=-45)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Radar chart for model comparison
            if len(df) <= 5:  # Only show radar for <= 5 models
                st.subheader("🎯 Model Performance Radar")
                fig_radar = go.Figure()

                for idx, row in df.iterrows():
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=[row["Test Accuracy"], row["Test F1"]],
                            theta=["Accuracy", "F1 Score"],
                            fill="toself",
                            name=row["Display Name"],
                            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.2%}<extra></extra>",
                        )
                    )

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=500,
                    title="Model Performance Profile",
                )

                st.plotly_chart(fig_radar, use_container_width=True)

        else:  # Regression metrics
            # Multi-metric comparison for regression
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("R² Score", "RMSE", "MAE"),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
            )

            # R² chart
            fig.add_trace(
                go.Bar(
                    x=df["Display Name"],
                    y=df["Test R²"],
                    name="R²",
                    marker_color="rgb(46, 204, 113)",
                    text=df["Test R²"].apply(lambda x: f"{x:.3f}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>",
                    customdata=df["Run Name"],
                    hovertext=df["Run Name"],
                ),
                row=1,
                col=1,
            )

            # RMSE chart (lower is better - use inverted color)
            fig.add_trace(
                go.Bar(
                    x=df["Display Name"],
                    y=df["Test RMSE"],
                    name="RMSE",
                    marker_color="rgb(231, 76, 60)",
                    text=df["Test RMSE"].apply(lambda x: f"{x:.3f}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>",
                    customdata=df["Run Name"],
                    hovertext=df["Run Name"],
                ),
                row=1,
                col=2,
            )

            # MAE chart
            fig.add_trace(
                go.Bar(
                    x=df["Display Name"],
                    y=df["Test MAE"],
                    name="MAE",
                    marker_color="rgb(52, 152, 219)",
                    text=df["Test MAE"].apply(lambda x: f"{x:.3f}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>",
                    customdata=df["Run Name"],
                    hovertext=df["Run Name"],
                ),
                row=1,
                col=3,
            )

            fig.update_layout(
                height=500,
                showlegend=False,
                title_text="Regression Metrics Comparison",
                title_x=0.5,
                hovermode="closest",
            )
            fig.update_xaxes(tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)

        # 📈 MODEL COMPARISON BY FRAMEWORK
        st.subheader("🔬 Model Performance by Framework")

        # Group by framework
        framework_colors = {
            "pytorch": "#EE4C2C",
            "sklearn": "#F7931E",
            "xgboost": "#0077B5",
        }

        if task_type == "classification":
            metric_col = "Test Accuracy"
        else:
            metric_col = "Test R²"

        fig_framework = px.scatter(
            df,
            x="Framework",
            y=metric_col,
            color="Model",
            size=[100] * len(df),
            hover_data=["Display Name", "Run Name", "Status"],
            title=f"{metric_col} Distribution by Framework",
            height=450,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )

        fig_framework.update_traces(
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )

        fig_framework.update_layout(
            xaxis_title="Framework",
            yaxis_title=metric_col,
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_framework, use_container_width=True)

        # 📋 ENHANCED DATA TABLE
        st.markdown("---")
        st.subheader("📋 Detailed Metrics Table")

        # Style the dataframe with color coding
        def highlight_best(s):
            if s.name in ["Test Accuracy", "Test F1", "Test R²"]:
                is_max = s == s.max()
                return ["background-color: #90EE90" if v else "" for v in is_max]
            elif s.name in ["Test MAE", "Test RMSE"]:
                is_min = s == s.min()
                return ["background-color: #90EE90" if v else "" for v in is_min]
            return [""] * len(s)

        styled_df = df.style.apply(
            highlight_best,
            subset=pd.IndexSlice[:, df.select_dtypes(include="number").columns],
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
        )

        # Detailed view
        st.divider()
        st.subheader("🔍 Detailed Run Information")

        # Create a mapping for selectbox
        display_to_full = {r["Display Name"]: r["Run Name"] for r in comparison_data}

        selected_display_name = st.selectbox(
            "Select a run to view details:",
            options=list(display_to_full.keys()),
            index=0,
            key=f"run_selector_{dataset_key}",
        )

        selected_run_name = display_to_full[selected_display_name]

        # Find the selected run
        selected_run = next(
            (r for r in runs if r.get("info", {}).get("run_name") == selected_run_name),
            None,
        )

        if selected_run:
            data = selected_run.get("data", {})
            info = selected_run.get("info", {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Metrics")
                metrics = {
                    metric["key"]: metric["value"] for metric in data.get("metrics", [])
                }
                for key, value in metrics.items():
                    if isinstance(value, float):
                        st.metric(key, f"{value:.4f}")
                    else:
                        st.metric(key, value)

            with col2:
                st.markdown("### ⚙️ Parameters")
                params = {
                    param["key"]: param["value"] for param in data.get("params", [])
                }
                for key, value in params.items():
                    st.text(f"{key}: {value}")

            # MLflow UI link
            st.divider()
            st.markdown(
                f"[🔗 View in MLflow UI]({MLFLOW_URL}/#/experiments/{info.get('experiment_id')}/runs/{info.get('run_id')})"
            )
    else:
        st.info("No run data available.")


# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigation", ["Upload Dataset", "Job Status", "Experiment Results"]
)

if page == "Upload Dataset":
    st.header("📤 Upload Dataset")

    with st.form("upload_form"):
        dataset_name = st.text_input("Dataset Name", "my_dataset")
        target_name = st.text_input("Target Column Name", "target")
        task_type = st.selectbox("Task Type", ["classification", "regression"])
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        submitted = st.form_submit_button("Start Training")

        if submitted and uploaded_file is not None:
            # Read CSV content
            try:
                df = pd.read_csv(uploaded_file)
                csv_content = df.to_csv(index=False)

                payload = {
                    "name": dataset_name,
                    "target_name": target_name,
                    "task_type": task_type,
                    "dataset_csv": csv_content,
                }

                with st.spinner("Sending dataset to API..."):
                    try:
                        response = requests.post(f"{API_URL}/d_dataset", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            st.success(
                                f"✅ Job started! Request ID: {data['request_id']}"
                            )
                            st.info(f"Track status in the 'Job Status' page.")

                            # Save request ID to session state to easily track it
                            if "recent_jobs" not in st.session_state:
                                st.session_state.recent_jobs = []
                            st.session_state.recent_jobs.insert(0, data["request_id"])
                        else:
                            st.error(f"❌ Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Could not connect to API. Is it running?")
            except Exception as e:
                st.error(f"Error processing file: {e}")

elif page == "Job Status":
    st.header("👀 Job Status Tracking")

    # Input for Request ID
    default_id = (
        st.session_state.recent_jobs[0]
        if "recent_jobs" in st.session_state and st.session_state.recent_jobs
        else ""
    )
    request_id = st.text_input("Enter Request ID", default_id)

    if st.button("Check Status") or request_id:
        if request_id:
            try:
                response = requests.get(f"{API_URL}/jobs/{request_id}")
                if response.status_code == 200:
                    status_data = response.json()

                    # Display Status with color
                    status = status_data.get("status", "UNKNOWN")
                    color = "blue"
                    if status == "COMPLETED":
                        color = "green"
                    elif status == "FAILED":
                        color = "red"
                    elif status == "QUEUED":
                        color = "orange"

                    st.markdown(f"### Status: :{color}[{status}]")
                    st.info(f"**Message:** {status_data.get('message', '')}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset:**", status_data.get("dataset_name"))
                    with col2:
                        st.write("**Created At:**", status_data.get("created_at"))

                    st.divider()
                    st.subheader("🏗️ Model Build Progress")

                    models = [
                        ("Linear Regression", "linear_regression"),
                        ("Random Forest", "random_forest"),
                        ("XGBoost", "xgboost"),
                        ("Neural Network", "feed_forward_nn"),
                    ]

                    model_cols = st.columns(2)
                    for i, (label, key) in enumerate(models):
                        with model_cols[i % 2]:
                            with st.container(border=True):
                                st.write(f"**{label}**")
                                progress_str = status_data.get(
                                    f"progress_{key}", "0/3"
                                )  # Default to 0/3 as most have 3
                                try:
                                    current, total = map(int, progress_str.split("/"))
                                    progress = min(current / total, 1.0)
                                except (ValueError, ZeroDivisionError):
                                    progress = 0.0
                                    current, total = 0, 3

                                # If job is completed, force 100%
                                if status == "COMPLETED":
                                    progress = 1.0
                                    st.progress(progress)
                                    st.caption("✅ Completed")
                                elif status == "FAILED":
                                    st.progress(progress)
                                    st.caption("❌ Failed")
                                else:
                                    st.progress(progress)
                                    if progress > 0:
                                        st.caption(f"Building... ({current}/{total})")
                                    else:
                                        st.caption("Waiting...")

                    # Show Job Map if available
                    if "job_map" in status_data:
                        st.divider()
                        st.subheader("📦 Built Docker Images")
                        job_map = json.loads(status_data["job_map"])
                        st.json(job_map)

                    # Auto-refresh button
                    if status not in ["COMPLETED", "FAILED"]:
                        time.sleep(1)
                        st.rerun()

                elif response.status_code == 404:
                    st.warning("Job not found.")
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to API.")

elif page == "Experiment Results":
    st.header("📊 Experiment Results")

    try:
        # Fetch experiments from MLflow - use search with proper filter
        experiments_response = requests.get(
            f"{MLFLOW_URL}/ajax-api/2.0/mlflow/experiments/search",
            params={"max_results": 1000},
        )

        if experiments_response.status_code == 200:
            experiments = experiments_response.json().get("experiments", [])

            if not experiments:
                st.info("No experiments found yet. Train some models first!")
            else:
                # Find the automl-experiments experiment
                automl_exp = next(
                    (exp for exp in experiments if exp["name"] == "automl-experiments"),
                    None,
                )

                if not automl_exp:
                    st.warning("No 'automl-experiments' found.")
                else:
                    exp_id = automl_exp["experiment_id"]

                    # Fetch runs for this experiment
                    runs_response = requests.post(
                        f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                        json={"experiment_ids": [exp_id], "max_results": 100},
                    )

                    if runs_response.status_code == 200:
                        runs = runs_response.json().get("runs", [])

                        if not runs:
                            st.info("No training runs found yet.")
                        else:
                            st.success(f"Found {len(runs)} training runs")

                            # Group runs by dataset
                            datasets = {}
                            for run in runs:
                                tags = run.get("data", {}).get("tags", [])
                                dataset_tag = next(
                                    (t for t in tags if t.get("key") == "dataset"), None
                                )
                                dataset = (
                                    dataset_tag.get("value", "Unknown")
                                    if dataset_tag
                                    else "Unknown"
                                )

                                if dataset not in datasets:
                                    datasets[dataset] = []
                                datasets[dataset].append(run)

                            # Create tabs for each dataset
                            if len(datasets) == 1:
                                dataset_name = list(datasets.keys())[0]
                                st.subheader(f"Dataset: {dataset_name}")
                                display_runs(
                                    datasets[dataset_name], dataset_key=dataset_name
                                )
                            else:
                                tabs = st.tabs(list(datasets.keys()))
                                for tab, (dataset_name, dataset_runs) in zip(
                                    tabs, datasets.items()
                                ):
                                    with tab:
                                        display_runs(
                                            dataset_runs, dataset_key=dataset_name
                                        )
                    else:
                        st.error("Failed to fetch runs from MLflow")
        else:
            st.error("Failed to connect to MLflow server")
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to MLflow. Make sure the server is running at http://localhost:5000"
        )
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
