
import json
def set_txt(username, lat, lon):

    data = {}
    data['result'] = []
    data['result'].append({
        'username': str(username),
        'latitude': str(lat),
        'longitude': str(lon)
    })

    f = open('staticFiles/output_data.txt', 'w')
    f.close()
    with open('staticFiles/output_data.txt', 'w') as outfile:
        json.dump(data, outfile)

def get_txt():
    with open('staticFiles/output_data.txt') as json_file:
        data = json.load(json_file)
        for p in data['result']:
            username = p['username']
            lat = p['latitude']
            lon = p['longitude']

    return username, lat, lon

def set_json(username, lat, lon):

    data = {}
    data['result'] = []
    data['result'].append({
        'username': str(username),
        'latitude': str(lat),
        'longitude': str(lon)
    })

    f = open('staticFiles/output_data.json', 'w')
    f.close()
    with open('staticFiles/output_data.json', 'w') as outfile:
        json.dump(data, outfile)

def get_json():
    with open('staticFiles/output_data.json') as json_file:
        data = json.load(json_file)
        for p in data['result']:
            username = p['username']
            lat = p['latitude']
            lon = p['longitude']

    return username, lat, lon


def set_csv(username, lat, lon):

    result_string = str(username) + ',' + str(lat) + ',' + str(lon)

    f = open('staticFiles/output_data.csv', 'w')
    f.close()
    with open('staticFiles/output_data.csv', 'w') as outfile:
        json.dump(result_string, outfile)

def get_csv():
    with open('staticFiles/output_data.csv') as json_file:
        data = json.load(json_file)
        for p in data.split(","):
            username = p[0]
            lat = p[1]
            lon = p[2]

    return username, lat, lon

