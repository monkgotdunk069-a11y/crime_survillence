"""
Pan-India Dataset Generator: Generates comprehensive crime surveillance dataset
spanning all 28 States and 8 Union Territories (36 Total) benchmarked against official NCRB publications.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# All 28 States and 8 Union Territories with representative cities and NCRB state weight shares (%)
PAN_INDIA_NCRB_STATE_DISTRIBUTION = {
    # 28 States
    'Uttar Pradesh': {'weight': 14.50, 'cities': ['Lucknow', 'Kanpur', 'Ghaziabad', 'Agra', 'Varanasi', 'Noida', 'Prayagraj', 'Meerut', 'Gorakhpur', 'Aligarh']},
    'Maharashtra': {'weight': 10.50, 'cities': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Kalyan', 'Vasai', 'Navi Mumbai', 'Aurangabad', 'Solapur']},
    'Madhya Pradesh': {'weight': 9.20, 'cities': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain', 'Sagar', 'Satna', 'Rewa']},
    'Rajasthan': {'weight': 7.80, 'cities': ['Jaipur', 'Jodhpur', 'Kota', 'Bikaner', 'Ajmer', 'Udaipur', 'Bhilwara', 'Alwar']},
    'Tamil Nadu': {'weight': 7.50, 'cities': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tiruppur', 'Erode', 'Vellore']},
    'Gujarat': {'weight': 6.20, 'cities': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar', 'Jamnagar', 'Gandhinagar']},
    'Kerala': {'weight': 5.80, 'cities': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Kollam', 'Thrissur', 'Kannur', 'Alappuzha']},
    'West Bengal': {'weight': 4.80, 'cities': ['Kolkata', 'Howrah', 'Asansol', 'Siliguri', 'Durgapur', 'Bardhaman', 'Darjeeling']},
    'Bihar': {'weight': 4.50, 'cities': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia', 'Darbhanga', 'Bihar Sharif']},
    'Karnataka': {'weight': 4.20, 'cities': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum', 'Gulbarga', 'Davanagere']},
    'Haryana': {'weight': 3.50, 'cities': ['Gurugram', 'Faridabad', 'Panipat', 'Ambala', 'Rohtak', 'Hisar', 'Karnal', 'Panchkula']},
    'Andhra Pradesh': {'weight': 3.30, 'cities': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Tirupati', 'Kakinada']},
    'Telangana': {'weight': 3.10, 'cities': ['Hyderabad', 'Secunderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam']},
    'Odisha': {'weight': 2.80, 'cities': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Berhampur', 'Sambalpur', 'Puri']},
    'Punjab': {'weight': 2.10, 'cities': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali', 'Pathankot']},
    'Jharkhand': {'weight': 1.80, 'cities': ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro', 'Deoghar', 'Hazaribagh']},
    'Chhattisgarh': {'weight': 1.70, 'cities': ['Raipur', 'Bhilai', 'Bilaspur', 'Korba', 'Durg', 'Jagdalpur']},
    'Assam': {'weight': 1.60, 'cities': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Nagaon', 'Tezpur']},
    'Uttarakhand': {'weight': 0.90, 'cities': ['Dehradun', 'Haridwar', 'Roorkee', 'Haldwani', 'Rishikesh', 'Nainital']},
    'Himachal Pradesh': {'weight': 0.50, 'cities': ['Shimla', 'Dharamshala', 'Solan', 'Mandi', 'Kullu', 'Manali']},
    'Tripura': {'weight': 0.35, 'cities': ['Agartala', 'Dharmanagar', 'Udaipur', 'Kailashahar']},
    'Goa': {'weight': 0.25, 'cities': ['Panaji', 'Margao', 'Vasco da Gama', 'Mapusa']},
    'Manipur': {'weight': 0.18, 'cities': ['Imphal', 'Churachandpur', 'Thoubal']},
    'Meghalaya': {'weight': 0.15, 'cities': ['Shillong', 'Tura', 'Jowai']},
    'Nagaland': {'weight': 0.12, 'cities': ['Kohima', 'Dimapur', 'Mokokchung']},
    'Arunachal Pradesh': {'weight': 0.10, 'cities': ['Itanagar', 'Pasighat', 'Tawang']},
    'Mizoram': {'weight': 0.08, 'cities': ['Aizawl', 'Lunglei', 'Champhai']},
    'Sikkim': {'weight': 0.06, 'cities': ['Gangtok', 'Namchi', 'Geyzing']},

    # 8 Union Territories
    'Delhi (NCT)': {'weight': 5.20, 'cities': ['Delhi', 'New Delhi', 'Dwarka', 'Rohini', 'Saket', 'Connaught Place', 'Karol Bagh']},
    'Jammu & Kashmir': {'weight': 0.80, 'cities': ['Srinagar', 'Jammu', 'Anantnag', 'Baramulla', 'Kathua', 'Udhampur']},
    'Chandigarh': {'weight': 0.30, 'cities': ['Chandigarh']},
    'Puducherry': {'weight': 0.20, 'cities': ['Puducherry', 'Pondicherry', 'Karaikal']},
    'Andaman and Nicobar Islands': {'weight': 0.07, 'cities': ['Port Blair']},
    'Dadra and Nagar Haveli and Daman and Diu': {'weight': 0.05, 'cities': ['Daman', 'Diu', 'Silvassa']},
    'Ladakh': {'weight': 0.04, 'cities': ['Leh', 'Kargil']},
    'Lakshadweep': {'weight': 0.02, 'cities': ['Kavaratti', 'Agatti']}
}

CRIME_DEFINITIONS = [
    # (Crime Description, Crime Domain, Default Weapon Dist, Probability Weight)
    ('BURGLARY', 'Other Crime', ['None', 'Blunt Object', 'Knife'], 0.18),
    ('THEFT', 'Other Crime', ['None'], 0.20),
    ('VEHICLE - STOLEN', 'Other Crime', ['None'], 0.12),
    ('ASSAULT', 'Violent Crime', ['Blunt Object', 'Knife', 'None', 'Firearm'], 0.12),
    ('ROBBERY', 'Violent Crime', ['Firearm', 'Knife', 'Blunt Object', 'None'], 0.07),
    ('DOMESTIC VIOLENCE', 'Violent Crime', ['Blunt Object', 'None', 'Knife', 'Poison'], 0.06),
    ('HOMICIDE', 'Violent Crime', ['Firearm', 'Knife', 'Blunt Object', 'Poison', 'Explosives'], 0.03),
    ('SEXUAL ASSAULT', 'Violent Crime', ['None', 'Knife', 'Blunt Object'], 0.025),
    ('KIDNAPPING', 'Violent Crime', ['Firearm', 'Knife', 'Blunt Object', 'None'], 0.02),
    ('FRAUD', 'Other Crime', ['None'], 0.04),
    ('CYBERCRIME', 'Other Crime', ['None'], 0.03),
    ('DRUG OFFENSE', 'Other Crime', ['None'], 0.025),
    ('EXTORTION', 'Violent Crime', ['Firearm', 'Knife', 'None'], 0.015),
    ('FIREARM OFFENSE', 'Violent Crime', ['Firearm'], 0.01),
    ('ARSON', 'Violent Crime', ['Explosives', 'Other', 'None'], 0.008),
    ('IDENTITY THEFT', 'Other Crime', ['None'], 0.01),
    ('COUNTERFEITING', 'Other Crime', ['None'], 0.007),
    ('ILLEGAL POSSESSION', 'Other Crime', ['Knife', 'Firearm', 'None'], 0.01),
    ('VANDALISM', 'Other Crime', ['Blunt Object', 'None'], 0.01),
    ('SHOPLIFTING', 'Other Crime', ['None'], 0.015),
    ('PUBLIC INTOXICATION', 'Other Crime', ['None'], 0.01),
    ('TRAFFIC VIOLATION', 'Traffic Fatality', ['None'], 0.04),
    ('FIRE ACCIDENT', 'Fire Accident', ['Explosives', 'None', 'Other'], 0.025)
]

PREMISES = [
    'Street/Sidewalk', 'Residential Apartment', 'Commercial Market', 
    'Public Transit / Metro Station', 'Bank / ATM Hub', 'Highway / Intersection',
    'Industrial Zone', 'Shopping Mall', 'Park / Public Ground', 'Government Facility'
]


def generate_pan_india_dataset(total_samples: int = 48000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Calculate state allocations
    states = list(PAN_INDIA_NCRB_STATE_DISTRIBUTION.keys())
    weights = [PAN_INDIA_NCRB_STATE_DISTRIBUTION[s]['weight'] for s in states]
    weights = np.array(weights) / sum(weights)
    
    state_sample_counts = np.random.multinomial(total_samples, weights)
    
    records = []
    report_num = 100001
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 7, 31)
    total_days = (end_date - start_date).days
    
    crime_names = [c[0] for c in CRIME_DEFINITIONS]
    crime_domains = {c[0]: c[1] for c in CRIME_DEFINITIONS}
    crime_weapons = {c[0]: c[2] for c in CRIME_DEFINITIONS}
    crime_probs = [c[3] for c in CRIME_DEFINITIONS]
    crime_probs = np.array(crime_probs) / sum(crime_probs)
    
    for state_idx, state_name in enumerate(states):
        n_samples = max(20, state_sample_counts[state_idx])
        cities = PAN_INDIA_NCRB_STATE_DISTRIBUTION[state_name]['cities']
        
        for _ in range(n_samples):
            # Select city
            city = np.random.choice(cities)
            
            # Select crime type
            crime_desc = np.random.choice(crime_names, p=crime_probs)
            crime_domain = crime_domains[crime_desc]
            
            # Select weapon
            avail_weapons = crime_weapons[crime_desc]
            weapon = np.random.choice(avail_weapons)
            
            # Generate date & time
            random_days = np.random.randint(0, total_days)
            occ_date = start_date + timedelta(days=random_days)
            
            # Hour with realistic night bias for violent crimes
            if crime_domain == 'Violent Crime':
                hour_p = np.array([
                    0.05, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03,
                    0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.07, 0.08,
                    0.08, 0.07, 0.06, 0.05
                ])
                hour = np.random.choice(24, p=hour_p / np.sum(hour_p))
            else:
                hour = np.random.randint(0, 24)
                
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            time_occ = f"{hour:02d}:{minute:02d}:{second:02d}"
            
            # Date reported
            delay_days = int(np.random.exponential(1.2))
            rep_date = occ_date + timedelta(days=delay_days)
            
            # Demographics
            victim_age = int(np.clip(np.random.normal(34, 14), 14, 88))
            victim_gender = np.random.choice(['M', 'F', 'X'], p=[0.68, 0.30, 0.02])
            
            # Premise
            premise = np.random.choice(PREMISES)
            
            # Police deployment (1 to 20 officers)
            police_deployed = int(np.random.choice(np.arange(1, 16), p=[
                0.22, 0.20, 0.16, 0.12, 0.09, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005
            ]))
            
            # Case closure (NCRB benchmark ~50%)
            case_closed = 'Yes' if np.random.rand() < 0.4996 else 'No'
            
            records.append({
                'Report Number': report_num,
                'Date Reported': rep_date.strftime('%Y-%m-%d'),
                'Date of Occurrence': occ_date.strftime('%Y-%m-%d'),
                'Time of Occurrence': time_occ,
                'City': city,
                'State': state_name,
                'Crime Description': crime_desc,
                'Crime Domain': crime_domain,
                'Victim Age': victim_age,
                'Victim Gender': victim_gender,
                'Victim Descent': 'Indian',
                'Premise Description': premise,
                'Weapon Used': weapon,
                'Police Deployed': police_deployed,
                'Case Closed': case_closed
            })
            
            report_num += 1
            
    df_out = pd.DataFrame(records)
    return df_out


if __name__ == '__main__':
    print("Generating comprehensive Pan-India Crime Dataset across all 36 States & UTs...")
    df_pan_india = generate_pan_india_dataset(total_samples=48000)
    df_pan_india.to_csv('crime_dataset_india.csv', index=False)
    print(f"Generated {len(df_pan_india):,} records spanning {df_pan_india['State'].nunique()} States/UTs and {df_pan_india['City'].nunique()} Cities!")
