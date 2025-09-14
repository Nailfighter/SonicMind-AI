from datetime import datetime
import json
import sys
import random
from audio import process_audio_file


def get_current_time():
    now = datetime.now()
    return now.strftime("%I:%M %p").lstrip("0")


def get_time_data():
    """Return time data as JSON for API consumption"""
    return {
        "time": get_current_time(),
        "timestamp": datetime.now().isoformat(),
        "message": "Hello from Python backend!",
        "function": "get_time_data"
    }


def get_random_number():
    """Return a random number with some metadata"""
    number = random.randint(1, 100)
    return {
        "random_number": number,
        "range": "1-100",
        "timestamp": datetime.now().isoformat(),
        "message": f"Random number generated: {number}",
        "function": "get_random_number"
    }


def get_system_info():
    """Return basic system information"""
    return {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
        "message": "System information retrieved",
        "function": "get_system_info"
    }


def get_weather_data():
    """Simulate weather data (mock)"""
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
    condition = random.choice(weather_conditions)
    temp = random.randint(-10, 35)
    
    return {
        "temperature": temp,
        "condition": condition,
        "humidity": random.randint(30, 90),
        "timestamp": datetime.now().isoformat(),
        "message": f"Weather: {temp}°C, {condition}",
        "function": "get_weather_data"
    }


def route_handler(route):
    """Route handler to call appropriate function based on route"""
    routes = {
        "time": get_time_data,
        "random": get_random_number,
        "system": get_system_info,
        "weather": get_weather_data
    }
    
    if route in routes:
        return routes[route]()
    else:
        return {
            "error": f"Unknown route: {route}",
            "available_routes": list(routes.keys()),
            "timestamp": datetime.now().isoformat(),
            "function": "route_handler"
        }


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "audio":
        # Audio processing mode: python routes.py audio <file_path> <filename> <process_type>
        audio_file_path = sys.argv[2]
        filename = sys.argv[3]
        process_type = sys.argv[4]
        result = process_audio_file(audio_file_path, filename, process_type)
        print(json.dumps(result))
    else:
        # Regular route mode: python routes.py <route>
        route = sys.argv[1] if len(sys.argv) > 1 else "time"
        result = route_handler(route)
        print(json.dumps(result))
