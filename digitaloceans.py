# DigitalOcean API with Python

import os
import time
import glob
import requests

# private constants
api_key = "1234567890"

# api url
url = "https://api.digitalocean.com/v2/droplets"


headers = {"Authorization": "Bearer {}".format(api_key)}
data = {
  "name": "HelloFromPython",
  "region": "nyc3",
  "size": "s-1vcpu-1gb",
  "image": "wordpress-20-04",
  "backups": False,
  "ipv6": False,
  "user_data": None,
  "private_networking": None,
  "volumes": None,
}
response = requests.post(url, data=data, headers=headers)
droplet_id = response.json()['droplet']['id']

time.sleep(60)

# get droplet ip address
droplet_url = "{}/{}".format(url, droplet_id)
droplet_response = requests.get(droplet_url, headers=headers)
ip_address = droplet_response.json()['droplet']['networks']['v4'][1]['ip_address']



# curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer 1234567890" -d '{
#   "name": "HelloFromTerminal",
#   "region": "nyc3",
#   "size": "s-1vcpu-1gb",
#   "image": "wordpress-20-04",
#   "backups": false,
#   "ipv6": false,
#   "user_data": null,
#   "private_networking": null,
#   "volumes": null,
#   "tags": [
#     "this is a tag"
#   ]
# }' "https://api.digitalocean.com/v2/droplets"