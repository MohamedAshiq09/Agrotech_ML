#!/usr/bin/env python3
"""
Test script for AgroGraphNet Prediction System
This script tests the prediction pipeline with the sample dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_prediction_system():
    """Test the prediction system with sample data"""

    try:
        from run_model import ModelPredictor

        print("🧪 Testing AgroGraphNet Prediction System...")
        print("=" * 50)

        # Initialize predictor
        print("1. Initializing predictor...")
        predictor = ModelPredictor()

        # Check if models are available
        if not predictor.gnn_models and not predictor.baseline_models:
            print("⚠️ No trained models found")
            print("   Run 05_model_development.ipynb first to train models")
            return False

        print(f"   Found {len(predictor.gnn_models)} GNN models and {len(predictor.baseline_models)} baseline models")

        # Test data loading
        print("\n2. Testing data loading...")
        sample_path = "sample_dataset.csv"

        if not os.path.exists(sample_path):
            print(f"❌ Sample dataset not found at {sample_path}")
            return False

        data_df = predictor.load_user_data(sample_path)
        print(f"   ✅ Loaded {len(data_df)} farms from sample dataset")

        # Test preprocessing
        print("\n3. Testing preprocessing...")
        graph_data, farm_ids = predictor.preprocess_for_prediction(data_df)
        print(f"   ✅ Created graph with {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")

        # Test prediction
        print("\n4. Testing predictions...")
        predictions = predictor.predict_with_all_models(graph_data, farm_ids)

        gnn_count = len([p for p in predictions['gnn_predictions'].values() if p is not None])
        baseline_count = len([p for p in predictions['baseline_predictions'].values() if p is not None])

        print(f"   ✅ Got predictions from {gnn_count} GNN models and {baseline_count} baseline models")

        if predictions['ensemble_prediction']:
            print(f"   ✅ Ensemble prediction completed for {len(predictions['ensemble_prediction']['predicted_classes'])} farms")

        # Test results display
        print("\n5. Testing results display...")
        results_df = predictor.display_results(predictions, data_df)

        if len(results_df) > 0:
            print(f"   ✅ Displayed results for {len(results_df)} farms")

        print("\n🎉 All tests passed! The prediction system is working correctly.")
        print("   You can now use 'python run_model.py' with your own datasets.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_system()
    sys.exit(0 if success else 1)
