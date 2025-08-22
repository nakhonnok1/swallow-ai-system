#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Quick API Test
"""

import requests
import json

def test_chat_api():
    url = "http://127.0.0.1:5000/api/chat"
    test_data = {"message": "สวัสดี AI Agent"}
    
    try:
        response = requests.post(url, json=test_data, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print("Error Response:")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat_api()
