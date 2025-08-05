import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from openai import OpenAI
import json
import time
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, data_path: str, openai_api_key: str = None):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.ml_models = {}
        self.ml_results = {}
        self.llm_results = {}
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset for ML models"""
        print("Loading and preprocessing data...")
        self.data = pd.read_csv(self.data_path)
        
        # Remove ID column and Date as they're not useful for prediction
        features_to_drop = ['id', 'Date']
        X = self.data.drop(features_to_drop + ['Price'], axis=1)
        y = self.data['Price']
        
        # Handle any missing values if present
        X = X.fillna(X.median())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for SVM and Linear Regression
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def train_ml_models(self):
        """Train traditional ML models"""
        print("\nTraining traditional ML models...")
        
        # Linear Regression
        self.ml_models['Linear Regression'] = LinearRegression()
        self.ml_models['Linear Regression'].fit(self.X_train_scaled, self.y_train)
        
        # Random Forest
        self.ml_models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.ml_models['Random Forest'].fit(self.X_train, self.y_train)
        
        # XGBoost
        self.ml_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.ml_models['XGBoost'].fit(self.X_train, self.y_train)
        
        # SVM Regressor
        self.ml_models['SVM'] = SVR(kernel='rbf', C=1000, gamma=0.001)
        self.ml_models['SVM'].fit(self.X_train_scaled, self.y_train)
        
    def evaluate_ml_models(self):
        """Evaluate traditional ML models"""
        print("\nEvaluating ML models...")
        
        for name, model in self.ml_models.items():
            if name in ['Linear Regression', 'SVM']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            self.ml_results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    def create_property_description(self, row):
        """Convert a property row to natural language description"""
        description = f"""Property Details:
- {int(row['number of bedrooms'])} bedrooms, {row['number of bathrooms']:.1f} bathrooms
- Living area: {int(row['living area'])} sq ft
- Lot area: {int(row['lot area'])} sq ft  
- {row['number of floors']:.1f} floors
- {'Waterfront property' if row['waterfront present'] else 'No waterfront'}
- {int(row['number of views'])} views
- Condition: {int(row['condition of the house'])}/10
- Grade: {int(row['grade of the house'])}/13
- House area (excluding basement): {int(row['Area of the house(excluding basement)'])} sq ft
- Basement area: {int(row['Area of the basement'])} sq ft
- Built in {int(row['Built Year'])}
- {'Renovated in ' + str(int(row['Renovation Year'])) if row['Renovation Year'] > 0 else 'Not renovated'}
- Postal code: {int(row['Postal Code'])}
- Location: ({row['Lattitude']:.4f}, {row['Longitude']:.4f})
- Renovated living area: {int(row['living_area_renov'])} sq ft
- Renovated lot area: {int(row['lot_area_renov'])} sq ft
- {int(row['Number of schools nearby'])} schools nearby
- {row['Distance from the airport']:.1f} km from airport"""
        return description
    
    def get_llm_prediction(self, property_description: str, examples: List[str] = None, shot_type: str = "zero"):
        """Get price prediction from LLM using different shot approaches"""
        
        if shot_type == "zero":
            prompt = f"""You are a real estate expert. Based on the property details below, estimate the price in Indian Rupees.

{property_description}

Respond with ONLY a number (the price estimate). Do not include currency symbols, commas, or explanations. For example: 2500000"""
        
        elif shot_type == "one":
            example = examples[0] if examples else ""
            prompt = f"""You are a real estate expert. Based on the property details, estimate the price in Indian Rupees.

Example:
{example}

Now estimate the price for this property:
{property_description}

Respond with ONLY a number (the price estimate). Do not include currency symbols, commas, or explanations."""
        
        else:  # few-shot
            examples_text = "\n\n".join(examples) if examples else ""
            prompt = f"""You are a real estate expert. Based on the property details, estimate the price in Indian Rupees.

Here are some examples:

{examples_text}

Now estimate the price for this property:
{property_description}

Respond with ONLY a number (the price estimate). Do not include currency symbols, commas, or explanations."""
        
        try:
            if not self.openai_client:
                print("OpenAI client not initialized")
                return None
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            prediction_text = response.choices[0].message.content.strip()
            print(f"LLM response: {prediction_text}")  # Debug output
            
            # Extract numerical value - improved parsing
            import re
            # Look for numbers (with optional commas and decimals)
            numbers = re.findall(r'[\d,]+\.?\d*', prediction_text.replace(',', ''))
            if numbers:
                # Take the largest number found (likely the price)
                prediction = float(max(numbers, key=lambda x: float(x.replace(',', ''))))
                return prediction
            else:
                print(f"No valid number found in: {prediction_text}")
                return None
            
        except Exception as e:
            print(f"Error getting LLM prediction: {e}")
            return None
    
    def prepare_few_shot_examples(self, n_examples: int = 5):
        """Prepare examples for few-shot learning"""
        # Use a subset of training data as examples
        example_indices = np.random.choice(len(self.X_train), n_examples, replace=False)
        examples = []
        
        for idx in example_indices:
            row = self.X_train.iloc[idx]
            price = self.y_train.iloc[idx]
            description = self.create_property_description(row)
            example = f"{description}\nPrice: ₹{price:,.0f}"
            examples.append(example)
        
        return examples
    
    def evaluate_llm_approaches(self, n_test_samples: int = 100):
        """Evaluate different LLM approaches"""
        print(f"\nEvaluating LLM approaches on {n_test_samples} samples...")
        
        # Sample test data to avoid excessive API costs
        test_indices = np.random.choice(len(self.X_test), min(n_test_samples, len(self.X_test)), replace=False)
        
        shot_types = {
            'Zero-shot': 'zero',
            'One-shot': 'one', 
            'Few-shot (5)': 'few',
            'Few-shot (10)': 'few',
            'Few-shot (20)': 'few'
        }
        
        for approach_name, shot_type in shot_types.items():
            print(f"\nEvaluating {approach_name}...")
            predictions = []
            actual_prices = []
            
            # Prepare examples based on approach
            if shot_type == 'one':
                examples = self.prepare_few_shot_examples(1)
            elif shot_type == 'few':
                if '5' in approach_name:
                    examples = self.prepare_few_shot_examples(5)
                elif '10' in approach_name:
                    examples = self.prepare_few_shot_examples(10)
                else:  # 20
                    examples = self.prepare_few_shot_examples(20)
            else:
                examples = None
            
            for idx in test_indices:
                row = self.X_test.iloc[idx]
                actual_price = self.y_test.iloc[idx]
                description = self.create_property_description(row)
                
                prediction = self.get_llm_prediction(description, examples, shot_type)
                
                if prediction is not None:
                    predictions.append(prediction)
                    actual_prices.append(actual_price)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            if predictions:
                predictions = np.array(predictions)
                actual_prices = np.array(actual_prices)
                
                mae = mean_absolute_error(actual_prices, predictions)
                mse = mean_squared_error(actual_prices, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_prices, predictions)
                
                self.llm_results[approach_name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'n_predictions': len(predictions)
                }
                
                print(f"{approach_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f} ({len(predictions)} predictions)")
    
    def compare_results(self):
        """Compare all approaches and display results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON: TRADITIONAL ML vs LLM APPROACHES")
        print("="*80)
        
        all_results = {**self.ml_results, **self.llm_results}
        
        # Sort by R² score (descending)
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        print(f"\n{'Approach':<20} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'Notes'}")
        print("-" * 70)
        
        for name, metrics in sorted_results:
            notes = ""
            if name in self.llm_results:
                notes = f"({metrics.get('n_predictions', 0)} samples)"
            
            print(f"{name:<20} {metrics['MAE']:<12.0f} {metrics['RMSE']:<12.0f} {metrics['R2']:<8.4f} {notes}")
        
        # Best performing approach
        best_approach = sorted_results[0]
        print(f"\nBest performing approach: {best_approach[0]} (R² = {best_approach[1]['R2']:.4f})")
        
        return all_results

def main():
    # Initialize predictor
    predictor = HousePricePredictor(
        data_path="/home/yurimarca/Code/fewshot-vs-ml-housing/data/House Price India.csv",
        openai_api_key=None  # Set your OpenAI API key here
    )
    
    # Load and preprocess data
    predictor.load_and_preprocess_data()
    
    # Train and evaluate ML models
    predictor.train_ml_models()
    predictor.evaluate_ml_models()
    
    # Evaluate LLM approaches (uncomment when you have OpenAI API key)
    # predictor.evaluate_llm_approaches(n_test_samples=50)
    
    # Compare all results
    results = predictor.compare_results()
    
    return results

if __name__ == "__main__":
    results = main()
