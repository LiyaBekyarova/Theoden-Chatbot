import requests
def get_real_weather(city="Varna"):
    try:
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_params = {"q": city, "format": "json", "limit": 1}
        headers = {'User-Agent': 'TheodenBot/1.0'}
        geo = requests.get(geo_url, params=geo_params, headers=headers, timeout=10).json()

        if not geo:
            return "No such land is known in the songs of the Mark."

        lat = geo[0]["lat"]
        lon = geo[0]["lon"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "auto"
        }
        data = requests.get(weather_url, params=params, timeout=10).json()
        w = data["current_weather"]

        temp_c = w["temperature"]
        temp_f = round(temp_c * 9/5 + 32)
        wind = w["windspeed"]
        code = w["weathercode"]

        descriptions = {
            (0,): "clear as the sky above Edoras",
            (1,): "mostly clear, the sun rides proud",
            (2,): "partly clouded, shadows drift across the plains",
            (3,): "overcast, the lid of heaven is shut",
            (45, 48): "fog lies thick in the vales",
            (51, 53, 55): "a light drizzle falls",
            (61, 63, 65): "rain beats upon shield and helm",
            (71, 73, 75, 77): "snow cloaks the earth in white silence",
            (80, 81, 82): "heavy rain lashes the Mark",
            (95, 96, 99): "thunder rolls and lightning splits the sky"
        }

        desc = "the heavens shift and murmur"
        for codes, description in descriptions.items():
            if code in codes:
                desc = description
                break

        return (f"In {city.title()}, the air stands at {temp_f}°F ({temp_c}°C), "
                f"wind {wind} km/h from the {w['winddirection']}° quarter. "
                f"The sky is {desc}.")

    except Exception as e:
        return "The ravens return empty-beaked. No tidings of weather reach Meduseld this hour."

