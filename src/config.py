import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
print(BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, 'models')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'