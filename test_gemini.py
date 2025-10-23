from google import genai
from dotenv import load_dotenv
import os

# Load the .env file from current directory
load_dotenv()


def test_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    print("API Key Loaded:", bool(api_key))  # should now print True
    if not api_key:
        print("❌ No API key found — check .env file location or variable name.")
        return

    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash-lite"

    try:
        response = client.models.generate_content(
            model=model_id,
            contents="Say hello from Gemini 2.5 Flash-Lite!"
        )
        print("✅ Gemini API is working!")
        print("Response:", response.text)
    except Exception as e:
        print("❌ Gemini API error:", e)


if __name__ == "__main__":
    test_gemini()
