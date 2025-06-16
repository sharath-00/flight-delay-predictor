import requests

mock_flights = {
    "AI101": {
        "airline": "Air India",
        "departure_code": "DEL",
        "arrival_code": "JFK",
        "scheduled_departure": "13:45"
    },
    "EK500": {
        "airline": "Emirates",
        "departure_code": "DXB",
        "arrival_code": "BOM",
        "scheduled_departure": "09:30"
    },
    "BA143": {
        "airline": "British Airways",
        "departure_code": "LHR",
        "arrival_code": "DEL",
        "scheduled_departure": "22:15"
    },
    "IG450": {
        "airline": "IndiGo",
        "departure_code": "CJB",
        "arrival_code": "MAA",
        "scheduled_departure": "18:20"
    },
    "QF402": {
        "airline": "Qantas",
        "departure_code": "SYD",
        "arrival_code": "MEL",
        "scheduled_departure": "07:00"
    },
    "UA808": {
        "airline": "United Airlines",
        "departure_code": "ORD",
        "arrival_code": "SFO",
        "scheduled_departure": "15:40"
    },
    "LH760": {
        "airline": "Lufthansa",
        "departure_code": "FRA",
        "arrival_code": "DEL",
        "scheduled_departure": "13:15"
    },
    "SQ421": {
        "airline": "Singapore Airlines",
        "departure_code": "BOM",
        "arrival_code": "SIN",
        "scheduled_departure": "23:55"
    },
    "DL450": {
        "airline": "Delta",
        "departure_code": "ATL",
        "arrival_code": "LAX",
        "scheduled_departure": "06:30"
    },
    "AF256": {
        "airline": "Air France",
        "departure_code": "CDG",
        "arrival_code": "BLR",
        "scheduled_departure": "14:50"
    },
    "SH2028":{
        
        "airline": "Shrinithi Dreams Airlines",
        "departure_code": "CJB",
        "arrival_code": "YYZ",
        "scheduled_departure": "14:50"

    },
    
        "MOS11":{
            "airline": "MOSCOW AIRLINES",
            "departure_code": "PEK",
            "arrival_code": "YQX",
            "scheduled_departure": "14:50"
        },
    
        "IG000":{
            "airline": "Indigo Airlines",
            "departure_code": "DEL",
            "arrival_code": "COK",
            "scheduled_departure": "14:50"
        }
    
}

def get_flight_info(flight_number):
    return mock_flights.get(flight_number.upper(), None)

def get_weather(airport_code, api_key):
    # Simulate bad weather for specific codes
    if airport_code == "DEL":
        return {
            'temperature': 35,
            'humidity': 70,
            'condition': "Hot and humid",
            'wind_kph': 10,
            'vis_km':23,
            'precip_mm': 0,
            'snow_cm': 0,
            'delay_reason': "Extreme heat"
        }
    elif airport_code == "JFK":
        return {
            'temperature': -15,
            'humidity': 85,
            'condition': "Snow",
            'wind_kph': 45,
            'vis_km': 0.5,
            'precip_mm': 10,
            'snow_cm': 5,
            'delay_reason': "Heavy snow and low visibility"
        }


def get_weather(airport_code, api_key):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={airport_code}"
        response = requests.get(url)
        data = response.json()

        location = data['location']
        current = data['current']

        return {
            'location_name': f"{location['name']}, {location['region']}, {location['country']}",
            'temperature': current['temp_c'],
            'humidity': current['humidity'],
            'condition': current['condition']['text'],
            'wind_kph': current['wind_kph'],
            'vis_km': current['vis_km'],
            'precip_mm': current['precip_mm'],
            'snow_cm': current.get('snow_cm', 0.0),
            'delay_reason': get_delay_reason(current)
        }
    except:
        return None###

def get_delay_reason(current):
    reasons = []
    if current.get('vis_km', 100) < 2:
        reasons.append("Low visibility")
    if current.get('wind_kph', 0) > 40:
        reasons.append("High wind speed")
    if current.get('precip_mm', 0) > 20:
        reasons.append("Heavy rain")
    if current.get('temp_c', 25) < -20:
        reasons.append("Extreme cold (de-icing)")
    if current.get('temp_c', 25) > 40:
        reasons.append("Extreme heat")
    if current.get('snow_cm', 0) > 2.5:
        reasons.append("Heavy snow")
    return ", ".join(reasons) if reasons else "Normal conditions"