import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from utils.paths import repo_root

env_loaded = False

def require_env_var(name: str) -> str:
    global env_loaded
    """Get an environment variable or exit with an error message."""
    if not env_loaded:
        load_env()
        env_loaded = True
    value = os.environ.get(name)
    if not value:
        print(f"Error: Environment variable '{name}' is missing or empty.")
        sys.exit(1)
    return value

def load_env():
    print("Loading environment")
    # Choose env file dynamically
    env = os.getenv("ENV", "ganache")  # defaults to dev if ENV not set
    env_file = repo_root(Path(__file__)) / ".env" / f".env.{env}"

    print(env_file)

    load_dotenv(env_file)

    print(f"Loaded environment: {env_file}")
