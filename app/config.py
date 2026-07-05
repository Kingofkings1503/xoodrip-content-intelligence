"""
config.py
---------
Centralized configuration for the entire application.

Uses pydantic-settings to automatically read values from environment
variables and from a `.env` file in the project root.

HOW IT WORKS (for Yash):
  - pydantic-settings looks at the `.env` file in the project root
  - It maps each line in `.env` to a field in the Settings class
  - Example: `MONGO_URI=mongodb+srv://...` in .env → `settings.MONGO_URI` in code
  - If the env var is not set, the default value from the class is used

Usage anywhere in the app:
    from app.config import settings
    print(settings.MONGO_URI)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB Atlas connection string — loaded from .env
    MONGO_URI: str = "mongodb://localhost:27017"

    # Database name inside MongoDB — all our collections live here
    MONGO_DB_NAME: str = "xoodrip_intelligence"

    # API key for authenticating /analyze/* requests
    XOODRIP_API_KEY: str = "dev-secret-key"

    class Config:
        # Tell pydantic-settings WHERE to find the .env file
        env_file = ".env"


# Create a single Settings instance — imported everywhere
settings = Settings()
