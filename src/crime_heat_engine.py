"""
Crime Heat Engine: Spatial & Temporal Hotspot Formulation & Crime Heat Index (CHI)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# ==============================================================================
# PAN-INDIA 36 STATES & UNION TERRITORIES + 250+ CITIES GEOGRAPHIC DATABASE
# ==============================================================================

STATE_CENTERS: Dict[str, Tuple[float, float]] = {
    # 28 States
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376),
    'Bihar': (25.0961, 85.3131),
    'Chhattisgarh': (21.2787, 81.8661),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2587, 71.1924),
    'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jharkhand': (23.6102, 85.2799),
    'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985),
    'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5122),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (18.1124, 79.0193),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550),

    # 8 Union Territories
    'Andaman and Nicobar Islands': (11.7401, 92.6586),
    'Chandigarh': (30.7333, 76.7794),
    'Dadra and Nagar Haveli and Daman and Diu': (20.4283, 72.8397),
    'Delhi (NCT)': (28.6139, 77.2090),
    'Delhi': (28.6139, 77.2090),
    'Jammu & Kashmir': (33.7782, 76.5762),
    'Ladakh': (34.1526, 77.5771),
    'Lakshadweep': (10.5667, 72.6417),
    'Puducherry': (11.9416, 79.8083)
}

# 250+ Indian Cities and District Headquarters
CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    # Punjab & Chandigarh
    'Chandigarh': (30.7333, 76.7794), 'Ludhiana': (30.9010, 75.8573), 'Amritsar': (31.6340, 74.8723),
    'Jalandhar': (31.3260, 75.5762), 'Patiala': (30.3398, 76.3869), 'Bathinda': (30.2110, 74.9455),
    'Mohali': (30.7046, 76.7179), 'SAS Nagar': (30.7046, 76.7179), 'Pathankot': (32.2689, 75.6499),
    'Hoshiarpur': (31.5143, 75.9115), 'Batala': (31.8186, 75.2028), 'Moga': (30.8230, 75.1734),
    'Abohar': (30.1453, 74.1993), 'Malerkotla': (30.5256, 75.8893), 'Khanna': (30.7071, 76.2163),
    'Phagwara': (31.2240, 75.7708), 'Firozpur': (30.9237, 74.6065),

    # Delhi (NCT)
    'Delhi': (28.6139, 77.2090), 'New Delhi': (28.6139, 77.2090), 'Dwarka': (28.5921, 77.0460),
    'Rohini': (28.7495, 77.0565), 'Saket': (28.5244, 77.2167), 'Connaught Place': (28.6315, 77.2167),
    'Karol Bagh': (28.6514, 77.1907), 'Lajpat Nagar': (28.5677, 77.2433), 'Janakpuri': (28.6219, 77.0878),

    # Uttar Pradesh
    'Lucknow': (26.8467, 80.9462), 'Kanpur': (26.4499, 80.3319), 'Ghaziabad': (28.6692, 77.4538),
    'Agra': (27.1767, 78.0081), 'Meerut': (28.9845, 77.7064), 'Varanasi': (25.3176, 82.9739),
    'Prayagraj': (25.4358, 81.8463), 'Allahabad': (25.4358, 81.8463), 'Bareilly': (28.3670, 79.4304),
    'Aligarh': (27.8974, 78.0880), 'Moradabad': (28.8353, 78.7747), 'Saharanpur': (29.9640, 77.5460),
    'Gorakhpur': (26.7606, 83.3732), 'Noida': (28.5355, 77.3910), 'Greater Noida': (28.4744, 77.5040),
    'Firozabad': (27.1591, 78.3957), 'Jhansi': (25.4484, 78.5685), 'Muzaffarnagar': (29.4727, 77.7085),
    'Mathura': (27.4924, 77.6737), 'Ayodhya': (26.7922, 82.1998), 'Faizabad': (26.7730, 82.1458),
    'Shahjahanpur': (27.8805, 79.9120), 'Rampur': (28.8154, 79.0257),

    # Haryana
    'Faridabad': (28.4089, 77.3178), 'Gurugram': (28.4595, 77.0266), 'Gurgaon': (28.4595, 77.0266),
    'Panipat': (29.3909, 76.9635), 'Ambala': (30.3782, 76.7767), 'Yamunanagar': (30.1290, 77.2674),
    'Rohtak': (28.8955, 76.6066), 'Hisar': (29.1492, 75.7217), 'Karnal': (29.6857, 76.9905),
    'Sonipat': (28.9931, 77.0151), 'Panchkula': (30.6942, 76.8606), 'Bhiwani': (28.7932, 76.1390),
    'Sirsa': (29.5349, 75.0289), 'Bahadurgarh': (28.6924, 76.9240), 'Jind': (29.3159, 76.3150),

    # Maharashtra
    'Mumbai': (19.0760, 72.8777), 'Pune': (18.5204, 73.8567), 'Nagpur': (21.1458, 79.0882),
    'Thane': (19.2183, 72.9781), 'Nashik': (19.9975, 73.7898), 'Kalyan': (19.2403, 73.1305),
    'Vasai': (19.3919, 72.8397), 'Virar': (19.4700, 72.8000), 'Navi Mumbai': (19.0330, 73.0297),
    'Aurangabad': (19.8762, 75.3433), 'Chhatrapati Sambhajinagar': (19.8762, 75.3433),
    'Solapur': (17.6599, 75.9064), 'Kolhapur': (16.7050, 74.2433), 'Amravati': (20.9320, 77.7523),
    'Nanded': (19.1383, 77.3210), 'Sangli': (16.8524, 74.5815), 'Jalgaon': (21.0077, 75.5626),
    'Akola': (20.7002, 77.0082), 'Latur': (18.4088, 76.5604), 'Dhule': (20.9042, 74.7749),
    'Ahmednagar': (19.0948, 74.7480), 'Chandrapur': (19.9615, 79.2961), 'Parbhani': (19.2644, 76.7748),

    # Karnataka
    'Bangalore': (12.9716, 77.5946), 'Bengaluru': (12.9716, 77.5946), 'Mysore': (12.2958, 76.6394),
    'Mysuru': (12.2958, 76.6394), 'Hubli': (15.3647, 75.1240), 'Hubballi': (15.3647, 75.1240),
    'Dharwad': (15.4589, 75.0078), 'Mangalore': (12.9141, 74.8560), 'Mangaluru': (12.9141, 74.8560),
    'Belgaum': (15.8497, 74.4977), 'Belagavi': (15.8497, 74.4977), 'Gulbarga': (17.3297, 76.8343),
    'Kalaburagi': (17.3297, 76.8343), 'Davanagere': (14.4644, 75.9218), 'Bellary': (15.1394, 76.9214),
    'Ballari': (15.1394, 76.9214), 'Bijapur': (16.8302, 75.7100), 'Vijayapura': (16.8302, 75.7100),
    'Shimoga': (13.9299, 75.5681), 'Shivamogga': (13.9299, 75.5681), 'Tumkur': (13.3392, 77.1017),
    'Udupi': (13.3409, 74.7421), 'Bidar': (17.9104, 77.5199), 'Hospet': (15.2689, 76.3909),

    # Tamil Nadu
    'Chennai': (13.0827, 80.2707), 'Coimbatore': (11.0168, 76.9558), 'Madurai': (9.9252, 78.1198),
    'Tiruchirappalli': (10.7905, 78.7047), 'Trichy': (10.7905, 78.7047), 'Salem': (11.6643, 78.1460),
    'Tiruppur': (11.1085, 77.3411), 'Erode': (11.3410, 77.7172), 'Vellore': (12.9165, 79.1325),
    'Tirunelveli': (8.7139, 77.7567), 'Thoothukudi': (8.7642, 78.1348), 'Tuticorin': (8.7642, 78.1348),
    'Dindigul': (10.3673, 77.9803), 'Thanjavur': (10.7870, 79.1378), 'Ranipet': (12.9229, 79.3326),
    'Nagercoil': (8.1833, 77.4119), 'Kanchipuram': (12.8342, 79.7036), 'Kumbakonam': (10.9602, 79.3845),

    # Telangana
    'Hyderabad': (17.3850, 78.4867), 'Secunderabad': (17.4399, 78.4983), 'Warangal': (17.9689, 79.5941),
    'Nizamabad': (18.6725, 78.0941), 'Khammam': (17.2473, 80.1514), 'Karimnagar': (18.4386, 79.1288),
    'Ramagundam': (18.7551, 79.5138), 'Mahbubnagar': (16.7488, 77.9928), 'Nalgonda': (17.0575, 79.2684),
    'Adilabad': (19.6641, 78.5320), 'Siddipet': (18.1018, 78.8520),

    # Gujarat
    'Ahmedabad': (23.0225, 72.5714), 'Surat': (21.1702, 72.8311), 'Vadodara': (22.3072, 73.1812),
    'Rajkot': (22.3039, 70.8022), 'Bhavnagar': (21.7645, 72.1519), 'Jamnagar': (22.4707, 70.0577),
    'Junagadh': (21.5222, 70.4579), 'Gandhinagar': (23.2156, 72.6369), 'Anand': (22.5645, 72.9289),
    'Navsari': (20.9500, 72.9300), 'Morbi': (22.8120, 70.8378), 'Nadiad': (22.6916, 72.8634),
    'Surendranagar': (22.7277, 71.6370), 'Bharuch': (21.7051, 72.9959), 'Mehsana': (23.5880, 72.3693),
    'Bhuj': (23.2420, 69.6669), 'Porbandar': (21.6417, 69.6293), 'Vapi': (20.3893, 72.9106),

    # Rajasthan
    'Jaipur': (26.9124, 75.7873), 'Jodhpur': (26.2389, 73.0243), 'Kota': (25.2138, 75.8648),
    'Bikaner': (28.0229, 73.3119), 'Ajmer': (26.4499, 74.6399), 'Udaipur': (24.5854, 73.7125),
    'Bhilwara': (25.3407, 74.6313), 'Alwar': (27.5530, 76.6346), 'Bharatpur': (27.2152, 77.5030),
    'Sikar': (27.6094, 75.1398), 'Pali': (25.7711, 73.3234), 'Sri Ganganagar': (29.9038, 73.8772),
    'Tonk': (26.1669, 75.7885), 'Beawar': (26.1011, 74.3214), 'Hanumangarh': (29.5818, 74.3168),

    # West Bengal
    'Kolkata': (22.5726, 88.3639), 'Howrah': (22.5958, 88.2636), 'Asansol': (23.6739, 86.9524),
    'Siliguri': (26.7271, 88.3953), 'Durgapur': (23.5204, 87.3119), 'Bardhaman': (23.2324, 87.8615),
    'Malda': (25.0108, 88.1411), 'Baharampur': (24.0988, 88.2685), 'Habra': (22.8456, 88.6322),
    'Kharagpur': (22.3460, 87.2320), 'Shantipur': (23.2500, 88.4300), 'Dankuni': (22.6800, 88.3000),
    'Darjeeling': (27.0410, 88.2663), 'Jalpaiguri': (26.5414, 88.7196),

    # Madhya Pradesh
    'Bhopal': (23.2599, 77.4126), 'Indore': (22.7196, 75.8577), 'Jabalpur': (23.1815, 79.9864),
    'Gwalior': (26.2183, 78.1828), 'Ujjain': (23.1765, 75.7885), 'Sagar': (23.8388, 78.7378),
    'Dewas': (22.9676, 76.0534), 'Satna': (24.5804, 80.8322), 'Ratlam': (23.3315, 75.0367),
    'Rewa': (24.5373, 81.3042), 'Murwara': (23.8343, 80.3995), 'Singrauli': (24.1997, 82.6644),
    'Burhanpur': (21.3106, 76.2298), 'Khandwa': (21.8314, 76.3498), 'Morena': (26.4947, 77.9940),

    # Bihar
    'Patna': (25.5941, 85.1376), 'Gaya': (24.7914, 85.0002), 'Bhagalpur': (25.2425, 86.9842),
    'Muzaffarpur': (26.1209, 85.3647), 'Purnia': (25.7771, 87.4753), 'Darbhanga': (26.1542, 85.8918),
    'Bihar Sharif': (25.1982, 85.5149), 'Arrah': (25.5541, 84.6663), 'Begusarai': (25.4182, 86.1272),
    'Katihar': (25.5541, 87.5722), 'Munger': (25.3757, 86.4744), 'Chhapra': (25.7796, 84.7499),
    'Danapur': (25.6333, 85.0333), 'Saharsa': (25.8835, 86.5986), 'Sasaram': (24.9535, 84.0298),

    # Andhra Pradesh
    'Visakhapatnam': (17.6868, 83.2185), 'Vijayawada': (16.5062, 80.6480), 'Guntur': (16.3067, 80.4365),
    'Nellore': (14.4426, 79.9865), 'Kurnool': (15.8281, 78.0373), 'Kakinada': (16.9891, 82.2475),
    'Rajahmundry': (17.0005, 81.8040), 'Kadapa': (14.4673, 78.8242), 'Tirupati': (13.6288, 79.4192),
    'Anantapur': (14.6819, 77.6006), 'Vizianagaram': (18.1067, 83.3956), 'Eluru': (16.7107, 81.0952),
    'Ongole': (15.5057, 80.0499), 'Nandyal': (15.4886, 78.4836), 'Machilipatnam': (16.1875, 81.1389),

    # Kerala
    'Thiruvananthapuram': (8.5241, 76.9366), 'Trivandrum': (8.5241, 76.9366), 'Kochi': (9.9312, 76.2673),
    'Cochin': (9.9312, 76.2673), 'Kozhikode': (11.2588, 75.7804), 'Calicut': (11.2588, 75.7804),
    'Kollam': (8.8932, 76.6141), 'Thrissur': (10.5276, 76.2144), 'Kannur': (11.8745, 75.3704),
    'Alappuzha': (9.4981, 76.3388), 'Alleppey': (9.4981, 76.3388), 'Kottayam': (9.5916, 76.5222),
    'Palakkad': (10.7867, 76.6548), 'Malappuram': (11.0732, 76.0740), 'Manjeri': (11.1215, 76.1210),

    # Odisha
    'Bhubaneswar': (20.2961, 85.8245), 'Cuttack': (20.4625, 85.8828), 'Rourkela': (22.2604, 84.8536),
    'Berhampur': (19.3149, 84.7941), 'Sambalpur': (21.4669, 83.9812), 'Puri': (19.8135, 85.8312),
    'Balasore': (21.4934, 86.9135), 'Bhadrak': (21.0544, 86.4955), 'Baripada': (21.9346, 86.7327),

    # Jharkhand
    'Ranchi': (23.3441, 85.3096), 'Jamshedpur': (22.8046, 86.2029), 'Dhanbad': (23.7957, 86.4304),
    'Bokaro': (23.6693, 86.1511), 'Deoghar': (24.4826, 86.7001), 'Phusro': (23.7712, 85.9922),
    'Hazaribagh': (23.9925, 85.3637), 'Giridih': (24.1860, 86.3097), 'Ramgarh': (23.6300, 85.5100),

    # Chhattisgarh
    'Raipur': (21.2514, 81.6296), 'Bhilai': (21.1938, 81.3509), 'Bilaspur': (22.0797, 82.1409),
    'Korba': (22.3595, 82.7501), 'Durg': (21.1904, 81.2849), 'Rajnandgaon': (21.0974, 81.0396),
    'Jagdalpur': (19.0748, 82.0081), 'Raigarh': (21.8974, 83.3950), 'Ambikapur': (23.1215, 83.1973),

    # Assam & North-East
    'Guwahati': (26.1445, 91.7362), 'Silchar': (24.8333, 92.7789), 'Dibrugarh': (27.4728, 94.9120),
    'Jorhat': (26.7509, 94.2037), 'Nagaon': (26.3464, 92.6840), 'Tinsukia': (27.4922, 95.3468),
    'Tezpur': (26.6338, 92.7926), 'Agartala': (23.8315, 91.2868), 'Shillong': (25.5788, 91.8933),
    'Aizawl': (23.7271, 92.7176), 'Imphal': (24.8170, 93.9368), 'Kohima': (25.6751, 94.1086),
    'Dimapur': (25.9094, 93.7266), 'Gangtok': (27.3389, 88.6065), 'Itanagar': (27.0844, 93.6053),

    # Uttarakhand & Himachal Pradesh
    'Dehradun': (30.3165, 78.0322), 'Haridwar': (29.9457, 78.1642), 'Roorkee': (29.8543, 77.8880),
    'Haldwani': (29.2183, 79.5130), 'Rudrapur': (28.9790, 79.4004), 'Rishikesh': (30.0869, 78.2676),
    'Nainital': (29.3919, 79.4542), 'Pauri': (30.1469, 78.7806), 'Pauri Garhwal': (30.1469, 78.7806),
    'Kotdwar': (29.7500, 78.5300), 'Almora': (29.6000, 79.6600), 'Pithoragarh': (29.5800, 80.2200),
    'Tehri': (30.3800, 78.4800), 'Chamoli': (30.4100, 79.3300), 'Uttarkashi': (30.7300, 78.4400),
    'Rudraprayag': (30.2800, 78.9800), 'Bageshwar': (29.8400, 79.7700), 'Champawat': (29.3300, 80.1000),
    'Kashipur': (29.2100, 78.9500), 'Ramnagar': (29.3900, 79.1200),
    'Shimla': (31.1048, 77.1734), 'Dharamshala': (32.2190, 76.3234),
    'Solan': (30.9045, 77.0967), 'Mandi': (31.5892, 76.9182), 'Kullu': (31.9579, 77.1095),
    'Manali': (32.2432, 77.1892), 'Hamirpur': (31.6862, 76.5213), 'Bilaspur (HP)': (31.3260, 76.7570),
    'Una': (31.4685, 76.2708), 'Chamba': (32.5534, 76.1258), 'Kangra': (32.0998, 76.2691),

    # Jammu & Kashmir, Ladakh & Islands
    'Srinagar': (34.0837, 74.7973), 'Jammu': (32.7266, 74.8570), 'Anantnag': (33.7311, 75.1522),
    'Baramulla': (34.2008, 74.3436), 'Kathua': (32.3716, 75.5186), 'Udhampur': (32.9262, 75.1416),
    'Leh': (34.1526, 77.5771), 'Kargil': (34.5539, 76.1349), 'Port Blair': (11.6234, 92.7265),
    'Kavaratti': (10.5667, 72.6417), 'Agatti': (10.8533, 72.1933), 'Silvassa': (20.2763, 73.0083),
    'Daman': (20.3974, 72.8328), 'Diu': (20.7144, 70.9874), 'Pondicherry': (11.9416, 79.8083),
    'Puducherry': (11.9416, 79.8083), 'Karaikal': (10.9254, 79.8380), 'Panaji': (15.4909, 73.8278),
    'Margao': (15.2832, 73.9862), 'Vasco da Gama': (15.3982, 73.8113), 'Mapusa': (15.5937, 73.8143),

    # All Remaining North-East & Island Towns
    'Kailashahar': (24.3320, 92.0059), 'Dharmanagar': (24.3686, 92.1644), 'Churachandpur': (24.3331, 93.6766),
    'Thoubal': (24.6397, 93.9999), 'Jowai': (25.4497, 92.2038), 'Tura': (25.5144, 90.2033),
    'Mokokchung': (26.3248, 94.5204), 'Tawang': (27.5861, 91.8594), 'Pasighat': (28.0664, 95.3268),
    'Lunglei': (22.8864, 92.7397), 'Champhai': (23.4754, 93.3275), 'Namchi': (27.1667, 88.3500),
    'Geyzing': (27.2833, 88.2500)
}

# Extensive State Aliases & Shortcodes
STATE_ALIASES: Dict[str, str] = {
    'panjab': 'Punjab', 'punjab': 'Punjab', 'pb': 'Punjab',
    'delhi': 'Delhi (NCT)', 'delhi nct': 'Delhi (NCT)', 'nct of delhi': 'Delhi (NCT)', 'nct': 'Delhi (NCT)',
    'up': 'Uttar Pradesh', 'u.p.': 'Uttar Pradesh', 'uttar pradesh': 'Uttar Pradesh', 'uttarpradesh': 'Uttar Pradesh',
    'mp': 'Madhya Pradesh', 'm.p.': 'Madhya Pradesh', 'madhya pradesh': 'Madhya Pradesh', 'madhyapradesh': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra', 'maharastra': 'Maharashtra', 'mh': 'Maharashtra',
    'karnataka': 'Karnataka', 'ka': 'Karnataka',
    'tamil nadu': 'Tamil Nadu', 'tamilnadu': 'Tamil Nadu', 'tn': 'Tamil Nadu',
    'west bengal': 'West Bengal', 'wb': 'West Bengal', 'bengal': 'West Bengal', 'paschim banga': 'West Bengal',
    'telangana': 'Telangana', 'ts': 'Telangana', 'tg': 'Telangana',
    'andhra pradesh': 'Andhra Pradesh', 'ap': 'Andhra Pradesh', 'andhra': 'Andhra Pradesh', 'andhrapradesh': 'Andhra Pradesh',
    'rajasthan': 'Rajasthan', 'raj': 'Rajasthan', 'rj': 'Rajasthan',
    'gujarat': 'Gujarat', 'gj': 'Gujarat',
    'kerala': 'Kerala', 'kl': 'Kerala',
    'bihar': 'Bihar', 'br': 'Bihar',
    'haryana': 'Haryana', 'hr': 'Haryana',
    'odisha': 'Odisha', 'orissa': 'Odisha', 'or': 'Odisha', 'od': 'Odisha',
    'jharkhand': 'Jharkhand', 'jh': 'Jharkhand',
    'chhattisgarh': 'Chhattisgarh', 'chhatisgarh': 'Chhattisgarh', 'cg': 'Chhattisgarh',
    'uttarakhand': 'Uttarakhand', 'uttaranchal': 'Uttarakhand', 'uk': 'Uttarakhand', 'ua': 'Uttarakhand',
    'himachal pradesh': 'Himachal Pradesh', 'himachal': 'Himachal Pradesh', 'hp': 'Himachal Pradesh',
    'jammu & kashmir': 'Jammu & Kashmir', 'jammu and kashmir': 'Jammu & Kashmir', 'j&k': 'Jammu & Kashmir', 'jk': 'Jammu & Kashmir',
    'ladakh': 'Ladakh', 'la': 'Ladakh',
    'assam': 'Assam', 'as': 'Assam',
    'goa': 'Goa', 'ga': 'Goa',
    'chandigarh': 'Chandigarh', 'ch': 'Chandigarh',
    'andaman and nicobar islands': 'Andaman and Nicobar Islands', 'andaman': 'Andaman and Nicobar Islands', 'an': 'Andaman and Nicobar Islands',
    'puducherry': 'Puducherry', 'pondicherry': 'Puducherry', 'py': 'Puducherry',
    'dadra and nagar haveli and daman and diu': 'Dadra and Nagar Haveli and Daman and Diu', 'daman': 'Dadra and Nagar Haveli and Daman and Diu', 'dnh': 'Dadra and Nagar Haveli and Daman and Diu',
    'lakshadweep': 'Lakshadweep', 'ld': 'Lakshadweep',
    'sikkim': 'Sikkim', 'sk': 'Sikkim',
    'tripura': 'Tripura', 'tr': 'Tripura',
    'meghalaya': 'Meghalaya', 'ml': 'Meghalaya',
    'manipur': 'Manipur', 'mn': 'Manipur',
    'mizoram': 'Mizoram', 'mz': 'Mizoram',
    'nagaland': 'Nagaland', 'nl': 'Nagaland',
    'arunachal pradesh': 'Arunachal Pradesh', 'arunachal': 'Arunachal Pradesh', 'ar': 'Arunachal Pradesh'
}

# Indian State / UT Mapping for known Cities
CITY_TO_STATE: Dict[str, str] = {
    # Punjab & Chandigarh
    'Chandigarh': 'Chandigarh', 'Ludhiana': 'Punjab', 'Amritsar': 'Punjab',
    'Jalandhar': 'Punjab', 'Patiala': 'Punjab', 'Bathinda': 'Punjab',
    'Mohali': 'Punjab', 'SAS Nagar': 'Punjab', 'Pathankot': 'Punjab',
    'Hoshiarpur': 'Punjab', 'Batala': 'Punjab', 'Moga': 'Punjab',
    'Abohar': 'Punjab', 'Malerkotla': 'Punjab', 'Khanna': 'Punjab',
    'Phagwara': 'Punjab', 'Firozpur': 'Punjab',

    # Delhi (NCT)
    'Delhi': 'Delhi (NCT)', 'New Delhi': 'Delhi (NCT)', 'Dwarka': 'Delhi (NCT)',
    'Rohini': 'Delhi (NCT)', 'Saket': 'Delhi (NCT)', 'Connaught Place': 'Delhi (NCT)',

    # Uttar Pradesh
    'Lucknow': 'Uttar Pradesh', 'Kanpur': 'Uttar Pradesh', 'Ghaziabad': 'Uttar Pradesh',
    'Agra': 'Uttar Pradesh', 'Meerut': 'Uttar Pradesh', 'Varanasi': 'Uttar Pradesh',
    'Prayagraj': 'Uttar Pradesh', 'Allahabad': 'Uttar Pradesh', 'Bareilly': 'Uttar Pradesh',
    'Aligarh': 'Uttar Pradesh', 'Moradabad': 'Uttar Pradesh', 'Saharanpur': 'Uttar Pradesh',
    'Gorakhpur': 'Uttar Pradesh', 'Noida': 'Uttar Pradesh', 'Greater Noida': 'Uttar Pradesh',
    'Firozabad': 'Uttar Pradesh', 'Jhansi': 'Uttar Pradesh', 'Muzaffarnagar': 'Uttar Pradesh',
    'Mathura': 'Uttar Pradesh', 'Ayodhya': 'Uttar Pradesh',

    # Haryana
    'Faridabad': 'Haryana', 'Gurugram': 'Haryana', 'Gurgaon': 'Haryana',
    'Panipat': 'Haryana', 'Ambala': 'Haryana', 'Yamunanagar': 'Haryana',
    'Rohtak': 'Haryana', 'Hisar': 'Haryana', 'Karnal': 'Haryana',
    'Sonipat': 'Haryana', 'Panchkula': 'Haryana',

    # Maharashtra
    'Mumbai': 'Maharashtra', 'Pune': 'Maharashtra', 'Nagpur': 'Maharashtra',
    'Thane': 'Maharashtra', 'Nashik': 'Maharashtra', 'Kalyan': 'Maharashtra',
    'Vasai': 'Maharashtra', 'Virar': 'Maharashtra', 'Navi Mumbai': 'Maharashtra',
    'Aurangabad': 'Maharashtra', 'Chhatrapati Sambhajinagar': 'Maharashtra',
    'Solapur': 'Maharashtra', 'Kolhapur': 'Maharashtra', 'Amravati': 'Maharashtra',
    'Nanded': 'Maharashtra', 'Sangli': 'Maharashtra', 'Jalgaon': 'Maharashtra',

    # Karnataka
    'Bangalore': 'Karnataka', 'Bengaluru': 'Karnataka', 'Mysore': 'Karnataka',
    'Mysuru': 'Karnataka', 'Hubli': 'Karnataka', 'Hubballi': 'Karnataka',
    'Mangalore': 'Karnataka', 'Mangaluru': 'Karnataka', 'Belgaum': 'Karnataka',
    'Belagavi': 'Karnataka', 'Gulbarga': 'Karnataka', 'Kalaburagi': 'Karnataka',
    'Davanagere': 'Karnataka', 'Bellary': 'Karnataka', 'Ballari': 'Karnataka',

    # Tamil Nadu
    'Chennai': 'Tamil Nadu', 'Coimbatore': 'Tamil Nadu', 'Madurai': 'Tamil Nadu',
    'Tiruchirappalli': 'Tamil Nadu', 'Salem': 'Tamil Nadu', 'Tiruppur': 'Tamil Nadu',
    'Erode': 'Tamil Nadu', 'Vellore': 'Tamil Nadu', 'Tirunelveli': 'Tamil Nadu',

    # Telangana
    'Hyderabad': 'Telangana', 'Secunderabad': 'Telangana', 'Warangal': 'Telangana',
    'Nizamabad': 'Telangana', 'Karimnagar': 'Telangana', 'Khammam': 'Telangana',

    # Gujarat
    'Ahmedabad': 'Gujarat', 'Surat': 'Gujarat', 'Vadodara': 'Gujarat',
    'Rajkot': 'Gujarat', 'Bhavnagar': 'Gujarat', 'Jamnagar': 'Gujarat',
    'Junagadh': 'Gujarat', 'Gandhinagar': 'Gujarat', 'Anand': 'Gujarat',

    # Rajasthan
    'Jaipur': 'Rajasthan', 'Jodhpur': 'Rajasthan', 'Kota': 'Rajasthan',
    'Bikaner': 'Rajasthan', 'Ajmer': 'Rajasthan', 'Udaipur': 'Rajasthan',
    'Bhilwara': 'Rajasthan', 'Alwar': 'Rajasthan', 'Bharatpur': 'Rajasthan',

    # West Bengal
    'Kolkata': 'West Bengal', 'Howrah': 'West Bengal', 'Asansol': 'West Bengal',
    'Siliguri': 'West Bengal', 'Durgapur': 'West Bengal', 'Darjeeling': 'West Bengal',

    # Madhya Pradesh
    'Bhopal': 'Madhya Pradesh', 'Indore': 'Madhya Pradesh', 'Jabalpur': 'Madhya Pradesh',
    'Gwalior': 'Madhya Pradesh', 'Ujjain': 'Madhya Pradesh', 'Sagar': 'Madhya Pradesh',
    'Satna': 'Madhya Pradesh', 'Rewa': 'Madhya Pradesh',

    # Bihar
    'Patna': 'Bihar', 'Gaya': 'Bihar', 'Bhagalpur': 'Bihar',
    'Muzaffarpur': 'Bihar', 'Purnia': 'Bihar', 'Darbhanga': 'Bihar', 'Bihar Sharif': 'Bihar',

    # Andhra Pradesh
    'Visakhapatnam': 'Andhra Pradesh', 'Vijayawada': 'Andhra Pradesh', 'Guntur': 'Andhra Pradesh',
    'Nellore': 'Andhra Pradesh', 'Kurnool': 'Andhra Pradesh', 'Tirupati': 'Andhra Pradesh', 'Kakinada': 'Andhra Pradesh',

    # Kerala
    'Thiruvananthapuram': 'Kerala', 'Kochi': 'Kerala', 'Kozhikode': 'Kerala',
    'Kollam': 'Kerala', 'Thrissur': 'Kerala', 'Kannur': 'Kerala', 'Alappuzha': 'Kerala',

    # Odisha
    'Bhubaneswar': 'Odisha', 'Cuttack': 'Odisha', 'Rourkela': 'Odisha', 'Puri': 'Odisha',
    'Sambalpur': 'Odisha', 'Berhampur': 'Odisha',

    # Jharkhand
    'Ranchi': 'Jharkhand', 'Jamshedpur': 'Jharkhand', 'Dhanbad': 'Jharkhand', 'Bokaro': 'Jharkhand',
    'Deoghar': 'Jharkhand', 'Hazaribagh': 'Jharkhand',

    # Chhattisgarh
    'Raipur': 'Chhattisgarh', 'Bhilai': 'Chhattisgarh', 'Bilaspur': 'Chhattisgarh', 'Durg': 'Chhattisgarh',
    'Korba': 'Chhattisgarh', 'Jagdalpur': 'Chhattisgarh',

    # North-East & Hill States
    'Guwahati': 'Assam', 'Dibrugarh': 'Assam', 'Silchar': 'Assam', 'Jorhat': 'Assam', 'Nagaon': 'Assam', 'Tezpur': 'Assam',
    'Shillong': 'Meghalaya', 'Tura': 'Meghalaya', 'Jowai': 'Meghalaya',
    'Imphal': 'Manipur', 'Churachandpur': 'Manipur', 'Thoubal': 'Manipur',
    'Aizawl': 'Mizoram', 'Lunglei': 'Mizoram', 'Champhai': 'Mizoram',
    'Kohima': 'Nagaland', 'Dimapur': 'Nagaland', 'Mokokchung': 'Nagaland',
    'Agartala': 'Tripura', 'Dharmanagar': 'Tripura', 'Kailashahar': 'Tripura',
    'Gangtok': 'Sikkim', 'Namchi': 'Sikkim', 'Geyzing': 'Sikkim',
    'Itanagar': 'Arunachal Pradesh', 'Pasighat': 'Arunachal Pradesh', 'Tawang': 'Arunachal Pradesh',
    'Dehradun': 'Uttarakhand', 'Haridwar': 'Uttarakhand', 'Rishikesh': 'Uttarakhand', 'Roorkee': 'Uttarakhand', 'Haldwani': 'Uttarakhand', 'Nainital': 'Uttarakhand',
    'Pauri': 'Uttarakhand', 'Pauri Garhwal': 'Uttarakhand', 'Kotdwar': 'Uttarakhand', 'Almora': 'Uttarakhand', 'Pithoragarh': 'Uttarakhand',
    'Tehri': 'Uttarakhand', 'Chamoli': 'Uttarakhand', 'Uttarkashi': 'Uttarakhand', 'Rudraprayag': 'Uttarakhand', 'Bageshwar': 'Uttarakhand', 'Champawat': 'Uttarakhand',
    'Shimla': 'Himachal Pradesh', 'Dharamshala': 'Himachal Pradesh', 'Manali': 'Himachal Pradesh', 'Solan': 'Himachal Pradesh', 'Mandi': 'Himachal Pradesh', 'Kullu': 'Himachal Pradesh', 'Hamirpur': 'Himachal Pradesh', 'Chamba': 'Himachal Pradesh',
    'Srinagar': 'Jammu & Kashmir', 'Jammu': 'Jammu & Kashmir', 'Anantnag': 'Jammu & Kashmir', 'Baramulla': 'Jammu & Kashmir', 'Kathua': 'Jammu & Kashmir', 'Udhampur': 'Jammu & Kashmir',
    'Leh': 'Ladakh', 'Kargil': 'Ladakh',
    'Panaji': 'Goa', 'Margao': 'Goa', 'Vasco da Gama': 'Goa', 'Mapusa': 'Goa',
    'Port Blair': 'Andaman and Nicobar Islands',
    'Puducherry': 'Puducherry', 'Pondicherry': 'Puducherry', 'Karaikal': 'Puducherry',
    'Kavaratti': 'Lakshadweep', 'Agatti': 'Lakshadweep',
    'Daman': 'Dadra and Nagar Haveli and Daman and Diu', 'Diu': 'Dadra and Nagar Haveli and Daman and Diu', 'Silvassa': 'Dadra and Nagar Haveli and Daman and Diu'
}

def resolve_city_state_coordinates(city: str, state: str = None) -> Tuple[float, float]:
    """
    Dynamically resolves coordinates for any input city and state across India
    with exact, partial, phonetic, and state-fallback matching.
    """
    city_clean = city.strip().title() if city else ''
    
    # 1. Exact match in city coordinates
    if city_clean in CITY_COORDINATES:
        return CITY_COORDINATES[city_clean]
    
    # 2. Case-insensitive / Substring match in city coordinates
    city_lower = city.strip().lower() if city else ''
    for c, coords in CITY_COORDINATES.items():
        if c.lower() == city_lower or c.lower() in city_lower or (len(city_lower) >= 3 and city_lower in c.lower()):
            return coords

    # 3. Check if city name is a state name (e.g. Chandigarh, Goa, Delhi)
    if city_lower in STATE_ALIASES:
        std_state = STATE_ALIASES[city_lower]
        if std_state in STATE_CENTERS:
            return STATE_CENTERS[std_state]

    # 4. Resolve State coordinates if state provided
    if state:
        state_lower = state.strip().lower()
        # Resolve via alias
        std_state = STATE_ALIASES.get(state_lower, None)
        if std_state and std_state in STATE_CENTERS:
            return STATE_CENTERS[std_state]
        
        # Substring match on State Centers
        for s, coords in STATE_CENTERS.items():
            if s.lower() in state_lower or state_lower in s.lower():
                return coords

    # 5. Default fallback to central India
    return (22.5000, 79.0000)

# Indian Penal Code (IPC) Crime Severity Weightings (0.0 to 1.0 scale)
CRIME_SEVERITY_WEIGHTS: Dict[str, float] = {
    'HOMICIDE': 1.00,
    'SEXUAL ASSAULT': 0.95,
    'KIDNAPPING': 0.90,
    'ROBBERY': 0.85,
    'FIREARM OFFENSE': 0.85,
    'ASSAULT': 0.80,
    'ARSON': 0.75,
    'DOMESTIC VIOLENCE': 0.70,
    'EXTORTION': 0.65,
    'BURGLARY': 0.60,
    'DRUG OFFENSE': 0.55,
    'ILLEGAL POSSESSION': 0.50,
    'FRAUD': 0.45,
    'COUNTERFEITING': 0.45,
    'IDENTITY THEFT': 0.40,
    'CYBERCRIME': 0.40,
    'VEHICLE - STOLEN': 0.35,
    'VANDALISM': 0.30,
    'SHOPLIFTING': 0.25,
    'PUBLIC INTOXICATION': 0.20,
    'TRAFFIC VIOLATION': 0.15
}

# Weapon Hazard Multiplier
WEAPON_WEIGHTS: Dict[str, float] = {
    'Explosives': 1.5,
    'Firearm': 1.4,
    'Knife': 1.25,
    'Poison': 1.2,
    'Blunt Object': 1.15,
    'Other': 1.05,
    'None': 1.0
}


def compute_temporal_risk_factor(hour: int, is_weekend: int) -> float:
    """
    Computes diurnal and weekly temporal risk factor based on empirical crime clocks.
    Peak danger hours: 20:00 - 03:00 (Night/Late Night) and Weekends.
    """
    if 20 <= hour <= 23 or 0 <= hour <= 3:
        diurnal_factor = 1.35
    elif 17 <= hour <= 19:
        diurnal_factor = 1.15
    elif 12 <= hour <= 16:
        diurnal_factor = 1.00
    else:  # Morning (04:00 - 11:00)
        diurnal_factor = 0.85

    weekend_factor = 1.12 if is_weekend else 1.0
    return float(diurnal_factor * weekend_factor)


def preprocess_crime_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, parses, geocodes, and enriches raw crime surveillance data
    with Spatio-temporal features and Crime Heat Index (CHI).
    """
    df = df_raw.copy()

    # 1. Parse Dates
    df['Date_Reported'] = pd.to_datetime(df['Date Reported'], format='mixed', errors='coerce')
    df['Date_Occ'] = pd.to_datetime(df['Date of Occurrence'], format='mixed', errors='coerce')
    df['Time_Occ'] = pd.to_datetime(df['Time of Occurrence'], format='mixed', errors='coerce')

    # Fill any null parsed dates with forward fill
    df['Date_Occ'] = df['Date_Occ'].ffill()
    df['Date_Reported'] = df['Date_Reported'].fillna(df['Date_Occ'])

    # 2. Extract Temporal Features
    df['Hour'] = df['Date_Occ'].dt.hour
    df['DayOfWeek'] = df['Date_Occ'].dt.dayofweek
    df['DayName'] = df['Date_Occ'].dt.day_name()
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['Month'] = df['Date_Occ'].dt.month
    df['Year'] = df['Date_Occ'].dt.year

    def get_time_of_day(hour: int) -> str:
        if 0 <= hour < 5:
            return 'Late Night'
        elif 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)

    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    df['Season'] = df['Month'].apply(get_season)

    # 3. Response & Case Dynamics
    df['ReportingDelay_Hours'] = (df['Date_Reported'] - df['Date_Occ']).dt.total_seconds() / 3600.0
    df['ReportingDelay_Hours'] = df['ReportingDelay_Hours'].apply(lambda x: max(0.0, x) if pd.notnull(x) else 0.0)

    # Weapon cleanup
    df['Weapon_Clean'] = df['Weapon Used'].fillna('None').astype(str)

    # 4. Geocoding & Micro-Precinct Spatial Jittering (Realistic Urban Distribution)
    df['State'] = df['City'].map(CITY_TO_STATE).fillna('Other State')
    coords = df['City'].map(CITY_COORDINATES)
    base_lats = [c[0] if isinstance(c, (tuple, list)) else 20.5937 for c in coords]
    base_lons = [c[1] if isinstance(c, (tuple, list)) else 78.9629 for c in coords]

    # Generate micro-zone offset within ~5-8km radius of city center
    np.random.seed(42)
    # Deterministic pseudo-random jitter per report number
    rep_num = df['Report Number'].fillna(0).astype(int)
    lat_jitters = ((rep_num * 17) % 100 - 50) * 0.0012
    lon_jitters = ((rep_num * 31) % 100 - 50) * 0.0012

    df['Latitude'] = [b_lat + j_lat for b_lat, j_lat in zip(base_lats, lat_jitters)]
    df['Longitude'] = [b_lon + j_lon for b_lon, j_lon in zip(base_lons, lon_jitters)]

    # 5. Crime Heat Formulation (CHI - Crime Heat Index)
    # Weights based on IPC Severity, Weapon Hazard, Temporal Risk, and Police Deployment
    df['SeverityWeight'] = df['Crime Description'].map(CRIME_SEVERITY_WEIGHTS).fillna(0.5)
    df['WeaponHazard'] = df['Weapon_Clean'].map(WEAPON_WEIGHTS).fillna(1.0)
    
    # Temporal Risk Factor
    df['TemporalRisk'] = [compute_temporal_risk_factor(h, w) for h, w in zip(df['Hour'], df['IsWeekend'])]

    # Crime Heat Score calculation:
    # CHI = SeverityWeight * WeaponHazard * TemporalRisk * (1 + 10 / (Police_Deployed + 5))
    police_term = 1.0 + (10.0 / (df['Police Deployed'].clip(lower=1) + 5.0))
    df['CrimeHeatScore'] = df['SeverityWeight'] * df['WeaponHazard'] * df['TemporalRisk'] * police_term

    # Standardize Crime Heat Score to 0-100 scale for intuitive surveillance monitoring
    min_score = df['CrimeHeatScore'].min()
    max_score = df['CrimeHeatScore'].max()
    df['CrimeHeatIndex'] = ((df['CrimeHeatScore'] - min_score) / (max_score - min_score)) * 100.0

    # Categorize into Heat Levels for Predictive Surveillance
    # Low Heat (< 35), Medium Heat (35 - 65), High Heat (> 65)
    def categorize_heat(score: float) -> str:
        if score >= 60.0:
            return 'High Heat'
        elif score >= 35.0:
            return 'Medium Heat'
        else:
            return 'Low Heat'

    df['HeatLevel'] = df['CrimeHeatIndex'].apply(categorize_heat)
    df['IsViolentCrime'] = df['Crime Domain'].apply(lambda x: 1 if x == 'Violent Crime' else 0)
    df['IsCaseClosed'] = df['Case Closed'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    return df
