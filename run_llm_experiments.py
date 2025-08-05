#!/usr/bin/env python3
"""
Demo script to run LLM experiments for house price prediction.

Usage:
1. Set your OpenAI API key as an environment variable:
   export OPENAI_API_KEY="your-api-key-here"

2. Run the script:
   python run_llm_experiments.py

This script will run a small-scale experiment comparing zero-shot, one-shot, 
and few-shot learning approaches using OpenAI's GPT model.
"""

import os
import sys
from dotenv import load_dotenv
from ml_vs_llm_comparison import HousePricePredictor

load_dotenv()

def main():
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OpenAI API key not found!")
        print("Please set your API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nAlternatively, you can set it directly in the script (not recommended for production)")
        sys.exit(1)
    
    print("🚀 Starting ML vs LLM House Price Prediction Experiment")
    print("=" * 60)
    
    # Initialize predictor with API key
    predictor = HousePricePredictor(
        data_path="data/House Price India.csv",
        openai_api_key=api_key
    )
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    predictor.load_and_preprocess_data()
    
    # Train traditional ML models first
    print("\n🤖 Training traditional ML models...")
    predictor.train_ml_models()
    predictor.evaluate_ml_models()
    
    # Run LLM experiments (smaller scale for demo)
    print("\n🧠 Running LLM experiments...")
    print("Note: Using 20 test samples to minimize API costs")
    predictor.evaluate_llm_approaches(n_test_samples=20)
    
    # Compare all results
    print("\n📈 Final comparison:")
    results = predictor.compare_results()
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Experiment completed successfully!")
        print("\n💡 Tips for further experimentation:")
        print("- Increase n_test_samples for more robust evaluation")
        print("- Try different prompt engineering techniques")
        print("- Experiment with different OpenAI models (gpt-4, etc.)")
        print("- Combine ML predictions with LLM reasoning for hybrid approaches")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Check your API key and internet connection")
