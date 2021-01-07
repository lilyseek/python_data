import time
import requests
import RPi.GPIO as GPIO
import Adafruit_DHT as dht

GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.IN)
GPIO.setup(16, GPIO.OUT)


TOKEN = ""  # Put your TOKEN here
DEVICE_LABEL = ""  # Put your device label here 
VARIABLE_LABEL_1 = "temperature"  # Put your first variable label here
VARIABLE_LABEL_2 = "humidity"  # Put your second variable label here

VARIABLE = "led"
def build_payload(variable_1, variable_2):
    
    h, t = dht.read_retry(dht.DHT11, 14)
    

    value_1 = t
    value_2 = h

   
    payload = {variable_1: value_1,
               variable_2: value_2,
                }

    return payload


def post_request(payload):
    # Creates the headers for the HTTP requests
    url = "http://industrial.api.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url, DEVICE_LABEL)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Makes the HTTP requests
    status = 400
    attempts = 0
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)

    # Processes results
    print(req.status_code, req.json())
    if status >= 400:
        print("[ERROR] Could not send data after 5 attempts, please check \
            your token credentials and internet connection")
        return False

    print("[INFO] request made properly, your device is updated")
    return True

def get_var(device, variable):
    try:
        url="http://industrial.api.ubidots.com/"
        url=url + \
             "api/v1.6/devices/{0}/{1}/".format(device, variable)
        headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}
        req = requests.get(url=url, headers=headers)
        return req.json()['last_value']['value']
    except:
        pass


def main():
    payload = build_payload(
        VARIABLE_LABEL_1, VARIABLE_LABEL_2)

    print("[INFO] Attemping to send data")
    post_request(payload)
    print("[INFO] finished")
    
    led = int(get_var(DEVICE_LABEL, VARIABLE))
    print('led', led)
    if (led == 1):
        GPIO.output(16, True)
    else:
        GPIO.output(16, False)

if __name__ == '__main__':
    while (True):
        main()
        time.sleep(1)