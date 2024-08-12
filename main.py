# main.py
import requests

class API:
    BASE_URL = 'http://127.0.0.1:5544/'
    PLAYER_STATUS = BASE_URL + 'playback/status'
    PLAYER_STOP_CURRENT = BASE_URL + 'campaign/current/stop'
    PLAYER_SHOW = BASE_URL + 'playback/show'
    PLAYER_HIDE = BASE_URL + 'playback/hide'
    PLAYER_START = BASE_URL + 'playback/start'
    PLAYER_STOP = BASE_URL + 'playback/stop'
    PLAYER_TRIGGER_CAMPAIGN = BASE_URL + 'trigger/1'





def player(url):
    """
    Makes a POST request to the given URL without sending any data.

    Args:
        url (str): The API endpoint URL.

    Returns:
        response (dict): The JSON response from the API if the request is successful.
        None: If the request fails.
    """
    try:


        # Make the POST request without sending any data
        response = requests.post(url)

        # Check the status code of the response
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print('Response content is not valid JSON')
                print('Response text:', response.text)
                return None
        else:
            print(f'Failed: {response.status_code} {response.text}')
            return None

    except Exception as e:
        print(f'An error occurred: {e}')
        return None



#  usage:
response_1 = player(API.PLAYER_STATUS)
response_2 = player(API.PLAYER_STOP_CURRENT)
# response_3 = player(API.PLAYER_STOP)

print('Response 1:', response_1)
print('Response 2:', response_2)
# print('Response 3:', response_3)