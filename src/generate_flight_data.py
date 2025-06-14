import pandas as pd
import random

airlines = ['Air India', 'IndiGo', 'SpiceJet', 'Emirates', 'British Airways']
airports = ['DEL', 'BOM', 'BLR', 'DXB', 'LHR', 'JFK']
weather_conditions = ['Sunny', 'Rain', 'Fog', 'Snow', 'Storm', 'Cloudy']

data = []

def generate_weather(extreme=False):
    if extreme:
        # Generate values likely to cause delays
        temp = random.choice([random.uniform(-30, -21), random.uniform(41, 46)])
        wind = random.randint(41, 60)
        visibility = random.randint(100, 700)
        rain = round(random.uniform(21, 30), 1)
        snow = round(random.uniform(2.6, 5.0), 1)
    else:
        # Normal weather values (no delay)
        temp = random.uniform(-10, 30)
        wind = random.randint(0, 30)
        visibility = random.randint(2000, 10000)
        rain = round(random.uniform(0, 10), 1)
        snow = round(random.uniform(0, 1), 1)

    condition = random.choice(weather_conditions)
    humidity = random.randint(30, 100)

    return temp, humidity, condition, wind, visibility, rain, snow

def generate_row(delay):
    airline = random.choice(airlines)
    dep_airport = random.choice(airports)
    arr_airport = random.choice([a for a in airports if a != dep_airport])
    scheduled_time = f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"

    # Generate weather
    dep_weather = generate_weather(extreme=delay)
    arr_weather = generate_weather(extreme=delay)

    delay_minutes = 0
    if delay:
        delay_minutes = random.randint(20, 120)

    return {
        'Airline': airline,
        'Dep_Airport': dep_airport,
        'Arr_Airport': arr_airport,
        'Scheduled_Departure': scheduled_time,
        'Dep_Temp': round(dep_weather[0], 1),
        'Dep_Humidity': dep_weather[1],
        'Dep_Weather': dep_weather[2],
        'Dep_WindSpeed': dep_weather[3],
        'Dep_Visibility': dep_weather[4],
        'Dep_RainfallRate': dep_weather[5],
        'Dep_SnowAccumulation': dep_weather[6],
        'Arr_Temp': round(arr_weather[0], 1),
        'Arr_Humidity': arr_weather[1],
        'Arr_Weather': arr_weather[2],
        'Arr_WindSpeed': arr_weather[3],
        'Arr_Visibility': arr_weather[4],
        'Arr_RainfallRate': arr_weather[5],
        'Arr_SnowAccumulation': arr_weather[6],
        'Delay_Minutes': delay_minutes,
        'Delay_Status': 1 if delay else 0
    }

# Generate 5000 delayed and 5000 on-time rows
for _ in range(5000):
    data.append(generate_row(delay=True))
    data.append(generate_row(delay=False))

df = pd.DataFrame(data)
df.to_csv("flight_weather_balanced.csv", index=False)
print("✅ Balanced dataset saved as flight_weather_balanced.csv")
