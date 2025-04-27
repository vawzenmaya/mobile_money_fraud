import requests
import json

# Local API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample transaction data based on your training features
sample_data = {
    "features": [
        720,       # step
        123456,    # initiator (encoded category)
        654321,    # recipient (encoded category)
        2,         # transactionType (encoded category)
        4.5,       # amount (log transformed)
        10000.0,   # oldBalInitiator
        9000.0,    # newBalInitiator
        5000.0,    # oldBalRecipient
        10000.0    # newBalRecipient
    ]
}

# Send POST request to your API
response = requests.post(url, json=sample_data)

# Print the response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=4))

# Try another sample (potentially fraudulent)
suspicious_sample = {
    "features": [
        360,       # step
        789012,    # initiator (encoded category)
        345678,    # recipient (encoded category)
        1,         # transactionType (encoded category)
        19824.96,       # amount (larger amount, log transformed)
        187712.18,   # oldBalInitiator
        167887.22,       # newBalInitiator (suspicious: balance went to zero)
        8.31,     # oldBalRecipient
        19833.27    # newBalRecipient (suspicious: large increase)
    ]
}

# Send POST request with suspicious data
print("\n--- Testing with suspicious transaction ---")
response = requests.post(url, json=suspicious_sample)

# Print the response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=4))

# 0,TRANSFER,19824.96,4537027967639631,187712.18,167887.22,4875702729424478,8.31,19833.27,1