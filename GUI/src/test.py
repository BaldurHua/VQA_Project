import requests
import json

url = "http://127.0.0.1:8000/vqa/"
headers = {"Content-Type": "application/json"}
data = {"image_base64": "test", "question": "what is this?"}

# Debugging output
print("Sending JSON Data:", json.dumps(data))

# Send the request
response = requests.post(url, headers=headers, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.text)