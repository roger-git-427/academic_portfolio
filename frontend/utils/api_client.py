import requests

class APIClient:
    def __init__(self, base_url="http://backend:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
    
    def get_data(self):
        """Get data from the backend API"""
        response = requests.get(f"{self.base_url}{self.api_prefix}/data")
        if response.status_code == 200:
            return response.json()
        return None

    def post_data(self, endpoint: str, data: dict):
        response = requests.post(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()

    def put_data(self, endpoint: str, data: dict):
        response = requests.put(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()

    def delete_data(self, endpoint: str):
        response = requests.delete(f"{self.base_url}/{endpoint}")
        response.raise_for_status()
        return response.json()