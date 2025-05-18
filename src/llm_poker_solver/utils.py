import os
from dotenv import load_dotenv


def get_hf_token():
    """
    Get the Hugging Face token from environment variables.
    Loads from .env file if available.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get token from environment
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN not found. Please create a .env file with your token or "
            "set the HF_TOKEN environment variable."
        )

    return token
