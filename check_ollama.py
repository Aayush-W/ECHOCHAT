#!/usr/bin/env python3
"""
Diagnostic script to check Ollama connection and model availability.
"""

import requests
import json
from pathlib import Path

OLLAMA_ENDPOINT = "http://localhost:11434"
MODEL_NAME = "mistral"

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is RUNNING")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is NOT RUNNING")
        print(f"   Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def list_available_models():
    """List all available models in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if models:
                print(f"\n✅ Available models ({len(models)}):")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size else 0
                    print(f"   - {name} ({size_gb:.1f} GB)")
            else:
                print("\n⚠️  No models installed!")
            return models
        return []
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_model_response(model_name="mistral"):
    """Test if a model can generate responses."""
    print(f"\n🔍 Testing {model_name} response generation...")
    
    url = f"{OLLAMA_ENDPOINT}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "Hi, how are you?",
        "stream": False,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            generated = result.get('response', '').strip()
            if generated:
                print(f"✅ Model responded successfully")
                print(f"   Response: {generated[:100]}...")
                return True
            else:
                print(f"❌ Model returned empty response")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out (>30s) - model might be too slow")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def main():
    print("=" * 60)
    print("OLLAMA DIAGNOSTIC CHECK")
    print("=" * 60)
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("\n💡 Start Ollama with: ollama serve")
        return
    
    # List available models
    models = list_available_models()
    
    # Check if required model is installed
    model_names = [m.get('name', '') for m in models]
    if MODEL_NAME in model_names:
        print(f"\n✅ Required model '{MODEL_NAME}' is installed")
    else:
        print(f"\n❌ Required model '{MODEL_NAME}' is NOT installed")
        print(f"   Install it with: ollama pull {MODEL_NAME}")
        return
    
    # Test model response
    if test_model_response(MODEL_NAME):
        print("\n✅ All checks passed! Ollama is ready.")
    else:
        print("\n❌ Model response test failed. Check logs for details.")

if __name__ == "__main__":
    main()
