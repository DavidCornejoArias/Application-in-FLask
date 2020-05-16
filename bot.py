import requests
import json

#fb_graph_url = 'https://graph.facebook.com/v2.7/me/messages'
fb_graph_url = 'https://graph.facebook.com/v6.0/me/messages?access_token=<EAAIRnbCD0ccBAOZCKuRjnekUwJcsXhwrTy14bDSwRbqyOP5wNSfhizpc7e7ZAN2p4yZA8ImJGX3IsjahJukL0gCTxYNOAorqd0raZC3GUZBUZCqMHpZCL87rXb0qQQLltAmaOkccme6nuiLBAQpQZBj8y75fxn1yKzofsCFOW3CjejbGZBmqYGnbaTZCZCC1kybGHgZD>'

class Bot(object):
    def __init__(self, access_token, api_url=fb_graph_url):
        self.access_token = access_token
        self.api_url = api_url

    def send_text_message(self, psid, message, messaging_type='RESPONSE'):
        headers = {
            'Content-Type': 'application/json'
            }

        data = {
            'messaging_type': messaging_type,
            'recipient': {'id': psid},
            'message': {'text': message}
            }
 
        params = {'access_token': self.access_token}
        #self.api_url = self.api_url + 'messages'
        response = requests.post(self.api_url,
                                headers=headers,
                                params=params,
                                data=json.dumps(data))
        print(response.content)
                                
#bot = Bot('EAAIRnbCD0ccBAOZCKuRjnekUwJcsXhwrTy14bDSwRbqyOP5wNSfhizpc7e7ZAN2p4yZA8ImJGX3IsjahJukL0gCTxYNOAorqd0raZC3GUZBUZCqMHpZCL87rXb0qQQLltAmaOkccme6nuiLBAQpQZBj8y75fxn1yKzofsCFOW3CjejbGZBmqYGnbaTZCZCC1kybGHgZD')
#bot.send_text_message(3129957067054763, 'Testing...')