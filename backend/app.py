from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import logging
import requests
import os
from vertexai import init
from vertexai.generative_models import GenerativeModel, SafetySetting
from google.oauth2.service_account import Credentials
from google.auth import load_credentials_from_file
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from typing import Dict, Any
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, pipeline


app = FastAPI()

origins = [
    "http://localhost:8501",  # Streamlit frontend
    "http://127.0.0.1:8501",  # Streamlit on localhost
    "http://localhost:3000",  # Example: React or other frontend
    "http://127.0.0.1:3000",
]
# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins],  # Replace "*" with your frontend URL for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH", "books-rag-41313ed7b44d.json")
if not CREDENTIALS_PATH:
    raise ValueError("CREDENTIALS_PATH environment variable is not set.")
if not os.path.isfile(CREDENTIALS_PATH):
    raise FileNotFoundError(f"Service account file not found at {CREDENTIALS_PATH}")
credentials = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
credentials.refresh(Request())


ENDPOINT = os.getenv("ENDPOINT", "https://us-central1-aiplatform.googleapis.com")
REGION = os.getenv("REGION", "us-central1")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_NAME = os.getenv("MODEL_NAME")


safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

def initialize_vertex_ai():
    init(project=PROJECT_ID, location=REGION)

@app.get("/")
def read_root():
    """
    Root endpoint providing basic information about the API and useful links.
    """
    return {
        "status": "Welcome to GYR backend microservice.",
        "description": "This microservice powers the GYR project by providing predictive modeling and risk assessment functionalities.",
        "documentation": {
            "docs_path": "/docs",
            "redoc_path": "/redoc"
        },
        "services": {
            "predictive_modeling": "/predict",
            "risk_assessment": "/risk"
        },
        "contact": "For issues or feedback, contact support@gyrproject.com"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "The GYR backend microservice is running properly."}

# # Load model and tokenizer
# @app.on_event("startup")
# def load_model():
#     global summarizer
#     model_dir = "./models/t5-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#     model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
#     summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

@app.on_event("startup")
def load_data():
    global final_df, merged_data, data
    try:
        data = pd.read_csv(os.getenv("DATA_PATH"))
        merged_data = pd.read_csv(os.getenv("MERGED_DATASET_PATH"))
        final_df = pd.read_csv(os.getenv("FINAL_DF_PATH"))

        # Convert date columns to datetime objects
        date_columns = [
            'projects_original_start_date', 'projects_original_end_date',
            'projects_actual_end_date', 'projects_approved_revised_end_date',
            'first_milestone_date', 'last_milestone_date'
        ]
        for col in date_columns:
            if col in merged_data.columns:
                merged_data[col] = pd.to_datetime(merged_data[col], errors='coerce')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV data: {e}")

# Request body model
class SummarizationRequest(BaseModel):
    text: str
    context: str

# Summarization endpoint
# @app.post("/v1/summarize")
# def summarize(request: SummarizationRequest):
#     try:
#         summary = summarizer(
#             request.text,
#             max_length=request.max_length,
#             min_length=request.min_length,
#             do_sample=False
#         )[0]["summary_text"]
#         return {"summary": summary}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/summarize")
def llm_summarize(request: SummarizationRequest) -> Dict[str, Any]:
    """
    Summarize text using the Gemini model.
    """
    if not request.text or not request.context:
        raise HTTPException(status_code=400, detail="Both 'text' and 'context' are required.")
    
    # Process the input
    full_prompt = f"Context:\n{request.context}\n\nUser Query:\n{request.text}\n\nGenerate a summary and actionable recommendations."
    try:
        initialize_vertex_ai()

        # Load the Gemini model
        model = GenerativeModel(MODEL_NAME)

        # Generate content using the Gemini model
        responses = model.generate_content(
            contents=[full_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        # Collect responses
        summaries = []
        for response in responses:
            summaries.append(response.text)

        return {"summary": "".join(summaries)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")
    
@app.get("/get_project_ids")
def get_project_ids():
    project_ids = data["project_id"].unique().tolist()
    return {"project_ids": project_ids}

@app.get("/get_dashboard_data")
def get_dashboard_data(project_id: int) -> Dict[str, Any]:
    try:
        # Example risk data with status added
        RISK_DATA = [
            {"risk_probability": 0.8, "risk_impact": 5, "risk_score": 4, "status": "open"},
            {"risk_probability": 0.6, "risk_impact": 3, "risk_score": 2, "status": "closed"},
            {"risk_probability": 0.9, "risk_impact": 4, "risk_score": 3.6, "status": "open"}
        ]

        # Convert risk data to DataFrame
        risk_df = pd.DataFrame(RISK_DATA)

        # Ensure required columns exist
        required_columns = ["risk_probability", "risk_impact", "risk_score", "status"]
        for col in required_columns:
            if col not in risk_df.columns:
                raise ValueError(f"Missing column '{col}' in risk data.")

        # Prepare other dashboard data
        response = {
            "risk_data": risk_df.to_dict(orient="records"),
            "status_counts": {"green": 10, "yellow": 5, "red": 2},
            "timeline_data": [
                {"planned_start": "2022-03-01", "planned_end": "2022-09-01", "actual_end": "2022-10-15"}
            ],
            "milestone_data": [
                {"milestone": "Milestone 1", "total_milestones": 5, "milestone_duration_days": 200, "milestone_completion_percentage": 80}
            ],
            "sentiment_data": [
                {"sentiment_date": "2023-01-01", "overall_gyr_status": "green", "open_issues_count": 3}
            ],
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing dashboard data: {str(e)}")

    # try:
    #     # Filter data by project_id
    #     project_data = data[data["project_id"] == project_id]
    #     if project_data.empty:
    #         raise HTTPException(status_code=404, detail="Project ID not found.")

    #     # Risk Data
    #     risk_data = project_data[["risk_impact", "risk_probability", "risk_score"]].rename(
    #         columns={
    #             "risk_probability": "settings_risk_probability",
    #             "risk_impact": "settings_risk_impact",
    #             "risk_score": "risks_avg_risk_score",
    #         }
    #     ).to_dict(orient="records")

    #     # Status Counts
    #     status_counts = {
    #         "green": int(project_data["status_green"].sum()),
    #         "yellow": int(project_data["status_yellow"].sum()),
    #         "red": int(project_data["status_red"].sum()),
    #     }

    #     # Timeline Data
    #     timeline_data = project_data[["planned_start", "planned_end", "actual_end"]].rename(
    #         columns={
    #             "planned_start": "project_original_start_date",
    #             "planned_end": "project_original_end_date",
    #             "actual_end": "project_actual_end_date",
    #         }
    #     ).to_dict(orient="records")

    #     # Milestone Data
    #     milestone_data = project_data[["total_milestones", "milestone_duration_days"]].rename(
    #         columns={
    #             "total_milestones": "hlm_total_milestones",
    #             "milestone_duration_days": "hlm_milestone_duration_days",
    #         }
    #     ).to_dict(orient="records")

    #     # Sentiment Data
    #     sentiment_data = project_data[["sentiment_date", "overall_gyr_status", "open_issues_count"]].to_dict(orient="records")

    #     # Issues Data
    #     issues_data = project_data[["issues_total", "issues_avg_priority", "issues_max_priority"]].rename(
    #         columns={
    #             "issues_total": "issues_total_issues",
    #         }
    #     ).to_dict(orient="records")

    #     # Deliverables Data
    #     deliverables_data = project_data[["deliverables_total", "deliverables_total_days", "deliverables_avg_duration"]].to_dict(orient="records")

    #     # Constraints Data
    #     constraints_data = project_data[["constraints_total", "constraints_status_trend"]].to_dict(orient="records")

    #     return {
    #         "risk_data": risk_data,
    #         "status_counts": status_counts,
    #         "timeline_data": timeline_data,
    #         "milestone_data": milestone_data,
    #         "sentiment_data": sentiment_data,
    #         "issues_data": issues_data,
    #         "deliverables_data": deliverables_data,
    #         "constraints_data": constraints_data,
    #     }
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error preparing dashboard data: {str(e)}")