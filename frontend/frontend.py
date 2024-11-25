import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(layout="wide", page_title="TBot Dashboard", page_icon="📊")

# Page Title
st.title("📊 TBot Dashboard")
st.write(f"Connecting to API at: {API_URL}")

# Fetch Project IDs
@st.cache_data
def fetch_project_ids():
    try:
        response = requests.get(f"{API_URL}/get_project_ids")
        response.raise_for_status()
        return response.json().get("project_ids", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching project IDs: {e}")
        return []

# Fetch Dashboard Data
@st.cache_data
def fetch_dashboard_data(project_id):
    try:
        response = requests.get(f"{API_URL}/get_dashboard_data", params={"project_id": project_id})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching dashboard data: {e}")
        return None

# Fetch available projects
project_ids = fetch_project_ids()

# Project dropdown
if project_ids:
    selected_project_id = st.selectbox("Select a Project ID", project_ids)
else:
    st.error("No project IDs available.")
    selected_project_id = None

# Fetch data for selected project
if selected_project_id:
    dashboard_data = fetch_dashboard_data(selected_project_id)
else:
    dashboard_data = None

if dashboard_data:
    # Key Metrics
    # Extract Metrics
    total_risks = len(dashboard_data.get("risk_data", []))
    avg_risk_score = (
        pd.DataFrame(dashboard_data.get("risk_data", []))["risk_score"].mean()
        if "risk_score" in pd.DataFrame(dashboard_data.get("risk_data", [])).columns
        else "N/A"
    )
    open_risks = sum(1 for risk in dashboard_data.get("risk_data", []) if risk.get("risk_status") == "open")
    milestones_completed = (
        sum(1 for milestone in dashboard_data.get("milestone_data", []) if milestone.get("milestone_completion_percentage") == 100)
    )
    budget_utilization = dashboard_data.get("budget_utilization", "$N/A")  # Replace with actual budget calculation if needed

    # Display Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Risks", total_risks)
    col2.metric("Average Risk Score", f"{avg_risk_score:.2f}" if avg_risk_score != "N/A" else "N/A")
    col3.metric("Open Risks", open_risks)
    col4.metric("Milestones Completed", milestones_completed)
    col5.metric("Budget Utilization", budget_utilization)

    # Filters
    st.sidebar.header("🔎 Filters")
    filter_risk_probability = st.sidebar.slider("Risk Probability", 0.0, 1.0, (0.0, 1.0))
    filter_risk_impact = st.sidebar.slider("Risk Impact", 0, 5, (0, 5))
    filter_status = st.sidebar.multiselect("Status", ["Green", "Yellow", "Red"], default=["Green", "Yellow", "Red"])

    # Apply filters to risk data
    risk_data = pd.DataFrame(dashboard_data.get("risk_data", []))
    if not risk_data.empty:
        filtered_risk_data = risk_data[
            (risk_data["risk_probability"].between(*filter_risk_probability)) &
            (risk_data["risk_impact"].between(*filter_risk_impact)) &
            (risk_data["status"].isin(filter_status))
        ]

        # Risk Heatmap
        st.header("🛠 Risk Heatmap")
        fig_heatmap = px.scatter(
            filtered_risk_data,
            x="risk_probability",
            y="risk_impact",
            size="risk_score",
            color="status",
            title="Filtered Risk Probability vs Impact",
            labels={"risk_probability": "Risk Probability", "risk_impact": "Risk Impact"},
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.write("No risk data available after applying filters.")

    # Timeline: Planned vs Actual
    timeline_data = pd.DataFrame(dashboard_data.get("timeline_data", []))
    if not timeline_data.empty:
        st.header("📅 Timeline: Planned vs Actual")
        fig_timeline = go.Figure()
        for _, row in timeline_data.iterrows():
            fig_timeline.add_trace(go.Scatter(
                x=[row["planned_start"], row["planned_end"]],
                y=["Planned Timeline", "Planned Timeline"],
                mode="lines",
                line=dict(color="blue", width=4),
                name="Planned"
            ))
            fig_timeline.add_trace(go.Scatter(
                x=[row["planned_start"], row["actual_end"]],
                y=["Actual Timeline", "Actual Timeline"],
                mode="lines",
                line=dict(color="red", width=4),
                name="Actual"
            ))
        fig_timeline.update_layout(title="Timeline: Planned vs Actual", xaxis_title="Date", yaxis_title="")
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.write("No timeline data available.")

    # Milestone Tracking
    milestone_data = pd.DataFrame(dashboard_data.get("milestone_data", []))
    if not milestone_data.empty:
        st.header("📋 Milestone Tracking")
        fig_milestones = px.bar(
            milestone_data,
            x="milestone",
            y="milestone_duration_days",
            color="milestone_completion_percentage",
            title="Milestone Completion",
            labels={"milestone": "Milestone", "milestone_duration_days": "Duration (days)"},
        )
        st.plotly_chart(fig_milestones, use_container_width=True)
    else:
        st.write("No milestone data available.")

    # Issue Tracking
    issue_data = pd.DataFrame(dashboard_data.get("issues_data", []))
    if not issue_data.empty:
        st.header("⚠️ Issue Tracking")
        fig_issues = px.bar(
            issue_data,
            x="issues_total",
            y="issues_avg_priority",
            title="Total Issues vs Priority",
            labels={"issues_total": "Total Issues", "issues_avg_priority": "Average Priority"},
        )
        st.plotly_chart(fig_issues, use_container_width=True)
    else:
        st.write("No issue data available.")

    # Chatbot Section
    st.subheader("💬 TBot Chat")
    if dashboard_data:
        project_status = ", ".join([f"{key}: {value}" for key, value in dashboard_data.get("status_counts", {}).items()])
        risk_summary = f"Total risks identified: {len(risk_data)}"
        timeline_summary = (
            f"Planned Start: {timeline_data['planned_start'].min() if not timeline_data.empty else 'N/A'}, "
            f"Planned End: {timeline_data['planned_end'].max() if not timeline_data.empty else 'N/A'}"
        )
        prompt_context = (
            f"Project ID: {selected_project_id}\n"
            f"Status: {project_status}\n"
            f"Risks: {risk_summary}\n"
            f"Timeline: {timeline_summary}\n\n"
            f"Please provide a detailed summary and recommendations for the project."
        )

        user_input = st.text_input("Ask a question or request recommendations:", key="chatbox")
        if st.button("Generate Insights"):
            try:
                payload = {"text": user_input, "context": prompt_context}
                response = requests.post(f"{API_URL}/v2/summarize", json=payload)
                response.raise_for_status()
                chatbot_response = response.json().get("summary", "No response.")
                st.write("**TBot Response:**")
                st.success(chatbot_response)
            except requests.exceptions.RequestException as e:
                st.error(f"Error interacting with TBot: {e}")

else:
    st.error("Failed to load dashboard data.")
