import requests

url = 'http://127.0.0.1:8000/predict'
data = {
    "vessel_class": ["Barge", "Fishing Vessel"],
    "build_year": [2005, 1970],
    "flag_abbr": ["US", "US"],
    "classification_society": ["UNSPECIFIED", "UNSPECIFIED"],
    "solas_desc": ["Non SOLAS", "Historical SOLAS"],
    "gross_ton": [764.0, 2749.0],
    "vessel_length": [200.0, 252.3]
}

response = requests.post(url, json=data)

print(response.json())
