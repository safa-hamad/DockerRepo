import pytest
from flask import Flask
from front_service import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_classify_audio_svm(client, mocker):
    mocker.patch('requests.post', return_value=MockResponse("Your music file type is label_svm"))
    response = client.post('/classify', data={'file': 'audio_data', 'model_selected': 'svm'})

    assert response.status_code == 200
    assert "Your music file type is label_svm" in response.get_data(as_text=True)

def test_classify_audio_vgg(client, mocker):
    mocker.patch('requests.post', return_value=MockResponse("Your music file type is label_vgg"))
    response = client.post('/classify', data={'file': 'audio_data', 'model_selected': 'vgg'})
    
    assert response.status_code == 200
    assert "Your music file type is label_vgg" in response.get_data(as_text=True)

def test_classify_audio_none(client):
    response = client.post('/classify', data={'file': 'audio_data', 'model_selected': 'none'})
    # assert response.status_code == 200