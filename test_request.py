import requests
import json

url = "http://127.0.0.1:5000/rag"
headers = {"Content-Type": "application/json"}
data = {"question": "한국 헌법 제1조는 무엇인가요?"}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    # Pretty-print the JSON response with proper decoding
    print("Response:", json.dumps(response.json(), ensure_ascii=False, indent=2))
else:
    print("Error:", response.status_code, response.text)
