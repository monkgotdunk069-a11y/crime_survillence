"""
End-to-End Training & NCRB Benchmark Validation Runner
"""

import json
import os
import pandas as pd
from src.crime_heat_engine import preprocess_crime_data
from src.ncrb_matcher import benchmark_against_ncrb
from src.ml_models import CrimeHeatPredictor


def main():
    print("=" * 70)
    print(" CRIME SURVEILLANCE HEAT ANALYSIS & NCRB BENCHMARK VALIDATION PIPELINE ")
    print("=" * 70)

    # 1. Load raw dataset
    dataset_path = 'crime_dataset_india.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"\n[1/5] Ingesting raw dataset from '{dataset_path}'...")
    df_raw = pd.read_csv(dataset_path)
    print(f"      Loaded {len(df_raw):,} records with {df_raw.shape[1]} features.")

    # 2. Preprocess, Geocode, and Formulate Crime Heat Index (CHI)
    print("\n[2/5] Preprocessing, geocoding 29 Indian cities, and calculating Crime Heat Index (CHI)...")
    df_processed = preprocess_crime_data(df_raw)
    
    # Save enriched dataset copy for fast loading / dashboard querying
    processed_path = 'crime_processed_enriched.csv'
    df_processed.to_csv(processed_path, index=False)
    print(f"      Enriched dataset saved to '{processed_path}'.")
    print("      Crime Heat Index Summary:")
    print(df_processed[['CrimeHeatIndex', 'HeatLevel', 'SeverityWeight', 'WeaponHazard']].describe())
    print("\n      Heat Level Breakdown:")
    print(df_processed['HeatLevel'].value_counts(normalize=True) * 100)

    # 3. Train ML Models
    print("\n[3/5] Training Predictive Surveillance Models (XGBoost Heat Classifier, Violent Risk, Police Optimizer)...")
    predictor = CrimeHeatPredictor(models_dir='models')
    ml_metrics = predictor.train_pipeline(df_processed)
    print(f"      Heat Classifier Accuracy: {ml_metrics['heat_accuracy']}%")
    print(f"      Heat Classifier ROC-AUC:  {ml_metrics['heat_roc_auc']}")
    print("      Top Feature Importances:")
    for feat, imp in list(ml_metrics['feature_importances'].items())[:6]:
        print(f"        - {feat}: {imp:.4f}")

    # 4. National Crime Records Bureau (NCRB) Benchmark Matching
    print("\n[4/5] Running Statistical NCRB Goodness-of-Fit & Empirical Distribution Audit...")
    ncrb_results = benchmark_against_ncrb(df_processed)
    print(f"      Composite NCRB Alignment Score: {ncrb_results['composite_ncrb_alignment_pct']}%")
    print(f"      Verdict:                        {ncrb_results['statistical_verdict']}")
    print(f"      City Share Cosine Similarity:   {ncrb_results['metrics']['city_distribution_cosine_similarity']}")
    print(f"      City Share MAPE Error:          {ncrb_results['metrics']['city_distribution_mape_pct']}%")
    print(f"      Chi-Square p-value:             {ncrb_results['metrics']['chi_square_p_value']}")
    print(f"      Case Closure Rate:              Predicted {ncrb_results['metrics']['case_closure_rate_predicted_pct']}% vs NCRB {ncrb_results['metrics']['case_closure_rate_ncrb_pct']}%")

    # Combine metrics into single JSON report
    combined_report = {
        'ml_performance': ml_metrics,
        'ncrb_validation': ncrb_results,
        'dataset_summary': {
            'total_incidents': len(df_processed),
            'unique_cities': df_processed['City'].nunique(),
            'crime_categories': df_processed['Crime Description'].nunique(),
            'date_span': f"{df_processed['Date_Occ'].min().strftime('%Y-%m-%d')} to {df_processed['Date_Occ'].max().strftime('%Y-%m-%d')}",
            'heat_distribution': {k: int(v) for k, v in df_processed['HeatLevel'].value_counts().items()}
        }
    }

    with open('models/ncrb_benchmarks.json', 'w') as f:
        json.dump(combined_report, f, indent=2)

    # 5. Quick Inference Demonstration
    print("\n[5/5] Running Sample Surveillance Hotspot Inference:")
    sample_incident = {
        'City': 'Delhi',
        'Crime Description': 'ROBBERY',
        'Victim Age': 28,
        'Victim Gender': 'M',
        'Weapon Used': 'Firearm',
        'Weapon_Clean': 'Firearm',
        'Hour': 22,
        'DayOfWeek': 5,
        'IsWeekend': 1,
        'Month': 10,
        'Police Deployed': 2
    }
    sample_pred = predictor.predict_surveillance_risk(sample_incident)
    print(f"      Input: {sample_incident['City']} | {sample_incident['Crime Description']} with {sample_incident['Weapon Used']} at {sample_incident['Hour']}:00 (Weekend)")
    print(f"      Predicted Heat Level:       {sample_pred['predicted_heat_level']}")
    print(f"      Threat Alert:               {sample_pred['threat_level']}")
    print(f"      Heat Class Probabilities:   {sample_pred['heat_probabilities']}")
    print(f"      Violent Crime Probability:  {sample_pred['violent_crime_probability_pct']}%")
    print(f"      Optimal Police Patrols:     {sample_pred['recommended_police_patrols']} units")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE: All models, metadata & NCRB verification successfully saved! ")
    print("=" * 70)


if __name__ == '__main__':
    main()
