import pytest
from flask import Flask, jsonify
from svm_service import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_svm_endpoint(client):
    data = [1, 2, 3]
    response = client.post('/svm', json=data)
    assert response.status_code == 200
    assert "Your music file type is" in response.get_data(as_text=True)
    predicted_label = response.get_data(as_text=True).split()[-1]
    assert predicted_label == "expected_label"

