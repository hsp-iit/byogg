import requests
import time

headers = {
    'Content-Type': 'application/json'
}

def startTobiiRecording(tobiiUri):
    print('before request', tobiiUri)
    response = requests.post(f'http://{tobiiUri}/rest/recorder!start', json=[], headers=headers)
    print('after request', response)
    timestamp = time.time()
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    return timestamp

def stopTobiiRecording(tobiiUri):
    response = requests.post(f'http://{tobiiUri}/rest/recorder!stop', json=[], headers=headers)
    timestamp = time.time()
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    return timestamp

def getRecordingFolder(tobiiUri):
    response = requests.get(f'http://{tobiiUri}/rest/recorder.folder', json=[], headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    return response.text

if __name__ == '__main__':
    tobiiUri = "yourUri" # INSERT TOBII GLASSES NAME HERE
    # Example of stream recording with tobii.
    timestamp = startRecording(tobiiUri)
    time.sleep(10)
    timestamp = stopRecording(tobiiUri)