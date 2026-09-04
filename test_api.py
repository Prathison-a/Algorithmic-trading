import requests

try:
    response = requests.post(
        "http://127.0.0.1:8000/forecast",
        json={
            "symbol": "SPY",
            "model": "TCN",
            "timeframe": 1,
            "forecast_steps": 15,
            "history": 500,
            "indicators": []
        }
    )
    print("STATUS:", response.status_code)
    if response.status_code != 200:
        print("BODY:", response.text)
except Exception as e:
    print(e)
