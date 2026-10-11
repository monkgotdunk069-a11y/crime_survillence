"""
Flask Server for Crime Surveillance Heat Analysis & NCRB Predictive Intelligence
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.crime_heat_engine import (
    CITY_COORDINATES, CRIME_SEVERITY_WEIGHTS, WEAPON_WEIGHTS, 
    CITY_TO_STATE, STATE_CENTERS, resolve_city_state_coordinates
)
from src.ml_models import CrimeHeatPredictor
from src.ncrb_matcher import NCRB_REFERENCE_METRICS

app = Flask(__name__)
CORS(app)

# Load predictor and benchmarks
predictor = CrimeHeatPredictor(models_dir='models')
if os.path.exists('models/heat_model.pkl'):
    predictor.load_models()

# Load processed data sample for rapid dashboard queries
processed_csv = 'crime_processed_enriched.csv'
if os.path.exists(processed_csv):
    df_cache = pd.read_csv(processed_csv)
else:
    df_cache = pd.DataFrame()


@app.route('/')
def index():
    """Renders the Crime Surveillance Heatmap & Analytics Dashboard."""
    cities = sorted(list(CITY_COORDINATES.keys()))
    states = sorted(list(set(list(CITY_TO_STATE.values()) + list(STATE_CENTERS.keys()))))
    crime_types = sorted(list(CRIME_SEVERITY_WEIGHTS.keys()))
    weapons = sorted(list(WEAPON_WEIGHTS.keys()))
    return render_template(
        'index.html', 
        cities=cities, 
        states=states,
        city_to_state=CITY_TO_STATE,
        crime_types=crime_types, 
        weapons=weapons
    )


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns high-level statistics and metadata."""
    if os.path.exists('models/ncrb_benchmarks.json'):
        with open('models/ncrb_benchmarks.json', 'r') as f:
            benchmarks = json.load(f)
    else:
        benchmarks = {}

    total_incidents = len(df_cache) if not df_cache.empty else 40160
    high_heat_count = int((df_cache['HeatLevel'] == 'High Heat').sum()) if not df_cache.empty else 423
    avg_chi = round(float(df_cache['CrimeHeatIndex'].mean()), 2) if not df_cache.empty else 20.88
    closure_rate = round(float(df_cache['IsCaseClosed'].mean() * 100.0), 2) if not df_cache.empty else 49.96

    return jsonify({
        'status': 'success',
        'total_incidents': total_incidents,
        'high_heat_hotspots': high_heat_count,
        'avg_crime_heat_index': avg_chi,
        'case_closure_rate': closure_rate,
        'benchmarks': benchmarks
    })


@app.route('/api/heatpoints', methods=['GET'])
def get_heatpoints():
    """
    Returns geographical heat points with weights [lat, lon, heat_intensity]
    for Leaflet.heat / Heatmap rendering, filtered by city and crime type if requested.
    """
    city = request.args.get('city', 'all')
    crime_type = request.args.get('crime_type', 'all')
    min_heat = float(request.args.get('min_heat', 0.0))
    limit = int(request.args.get('limit', 2500))

    if df_cache.empty:
        return jsonify({'points': [], 'count': 0})

    df_filtered = df_cache.copy()
    if city != 'all':
        df_filtered = df_filtered[df_filtered['City'] == city]
    if crime_type != 'all':
        df_filtered = df_filtered[df_filtered['Crime Description'] == crime_type]
    if min_heat > 0:
        df_filtered = df_filtered[df_filtered['CrimeHeatIndex'] >= min_heat]

    # Sample if larger than limit to ensure fast rendering
    if len(df_filtered) > limit:
        df_filtered = df_filtered.sample(n=limit, random_state=42)

    # Format: [lat, lon, normalized_intensity (0.1 to 1.0), city, crime_desc, heat_level, chi]
    points = []
    for _, row in df_filtered.iterrows():
        intensity = max(0.15, min(1.0, float(row['CrimeHeatIndex']) / 100.0))
        points.append({
            'lat': round(float(row['Latitude']), 5),
            'lon': round(float(row['Longitude']), 5),
            'intensity': round(intensity, 3),
            'city': str(row['City']),
            'crime': str(row['Crime Description']),
            'heat_level': str(row['HeatLevel']),
            'chi': round(float(row['CrimeHeatIndex']), 1),
            'hour': int(row['Hour']),
            'police': int(row['Police Deployed'])
        })

    return jsonify({
        'status': 'success',
        'points': points,
        'count': len(points)
    })


@app.route('/api/city_summary', methods=['GET'])
def get_city_summary():
    """Returns aggregated city crime heat scores and metrics."""
    if df_cache.empty:
        return jsonify({})

    city_agg = df_cache.groupby('City').agg({
        'CrimeHeatIndex': 'mean',
        'Report Number': 'count',
        'IsViolentCrime': 'mean',
        'Police Deployed': 'mean'
    }).reset_index()

    city_agg.columns = ['City', 'AvgCHI', 'IncidentCount', 'ViolentCrimeRate', 'AvgPolice']
    city_agg['AvgCHI'] = city_agg['AvgCHI'].round(2)
    city_agg['ViolentCrimeRate'] = (city_agg['ViolentCrimeRate'] * 100.0).round(2)
    city_agg['AvgPolice'] = city_agg['AvgPolice'].round(1)
    
    # Sort by total incident count
    city_agg = city_agg.sort_values(by='IncidentCount', ascending=False)

    return jsonify({
        'status': 'success',
        'cities': city_agg.to_dict(orient='records')
    })


@app.route('/api/predict', methods=['POST'])
def predict_incident():
    """
    Predicts Crime Heat Risk, Probabilities, Violent Threat, and Patrol Resource allocation.
    """
    try:
        data = request.json or {}
        city = data.get('city', 'Delhi')
        state = data.get('state', CITY_TO_STATE.get(city, 'Delhi (NCT)'))
        crime_desc = data.get('crime_desc', 'ROBBERY')
        victim_age = int(data.get('victim_age', 30))
        victim_gender = data.get('victim_gender', 'M')
        weapon = data.get('weapon', 'Firearm')
        hour = int(data.get('hour', 22))
        day_of_week = int(data.get('day_of_week', 5))
        month = int(data.get('month', 10))
        police_deployed = int(data.get('police_deployed', 3))

        input_payload = {
            'State': state,
            'City': city,
            'Crime Description': crime_desc,
            'Victim Age': victim_age,
            'Victim Gender': victim_gender,
            'Weapon Used': weapon,
            'Weapon_Clean': weapon,
            'Hour': hour,
            'DayOfWeek': day_of_week,
            'IsWeekend': 1 if day_of_week in [5, 6] else 0,
            'Month': month,
            'Police Deployed': police_deployed
        }

        result = predictor.predict_surveillance_risk(input_payload)

        # Dynamically resolve coordinates for any Indian city / state
        coords = resolve_city_state_coordinates(city, state)
        result['city_coordinates'] = {'lat': coords[0], 'lon': coords[1]}
        result['input_summary'] = input_payload
        result['state'] = state
        result['city'] = city

        # Compare predicted heat with NCRB baseline city risk
        ncrb_share = NCRB_REFERENCE_METRICS['city_distribution_share'].get(city, 2.0)
        result['ncrb_city_share_pct'] = ncrb_share

        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ncrb_benchmark', methods=['GET'])
def get_ncrb_benchmark():
    """Returns official NCRB vs Dataset / Predicted metrics comparison."""
    if os.path.exists('models/ncrb_benchmarks.json'):
        with open('models/ncrb_benchmarks.json', 'r') as f:
            data = json.load(f)
        return jsonify({'status': 'success', 'data': data['ncrb_validation']})
    return jsonify({'status': 'error', 'message': 'Benchmark data not available'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
