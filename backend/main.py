import uvicorn

if __name__ == "__main__":
    # The `app` module refers to the FastAPI application defined in `app.py`.
    uvicorn.run(
        "app:app",  # Format: "module_name:FastAPI_instance"
        host="127.0.0.1",  # Bind to all available network interfaces
        port=8000,       # Port on which the app will be accessible
        reload=True,     # Auto-reload on code changes (useful for development)
        workers=1        # Number of worker processes (can be increased in production)
    )
