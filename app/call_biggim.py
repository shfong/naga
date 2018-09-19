import json
import pandas
import requests
import time

base_url = 'http://biggim.ncats.io/api'

def get_table_from_biggim(data_name, threshold=0.5): 
    studies = get('metadata/study')
    study_names = [s['name'] for s in studies]
    tables = get('/metadata/table')
    default_table = [t for t in tables if t['default'] == True][0]['name']
    
    example_query = {
        "restriction_gt": f"{data_name},{threshold}",
        "table": default_table,
        "columns": "GTEx_Pancreas_Correlation",
        #"ids1": "3630,2645,5078,6927,6928,1056,8462,4760,3172,3651,6833,640,3767,26060",
        "limit": 400000000
    }
    
    try:
        query_submit = get('biggim/query', data=example_query)
        jprint(query_submit)
    except requests.HTTPError as e:
        print(e)

        jprint(e.response.json())


    try:
        while True:
            query_status = get('biggim/status/%s'% (query_submit['request_id'],))
            jprint(query_status)
            if query_status['status'] !='running':
                # query has finished
                break
            else:
                time.sleep(1)
                print("Checking again")
    except requests.HTTPError as e:
        print(e)

        jprint(e.response.json())

    result = pandas.concat(map(pandas.read_csv, query_status['request_uri']))

    return result 

#a couple of simple helper functions
def post(endpoint, data={}, base_url=base_url):
    req = requests.post('%s/%s' % (base_url,endpoint), data=data)
    req.raise_for_status()
    return req.json()

def get(endpoint, data={}, base_url=base_url):
    req = requests.get('%s/%s' % (base_url,endpoint), data=data)
    req.raise_for_status()
    print("Sent: GET %s?%s" % (req.request.url,req.request.body))
    return req.json()

def jprint(dct):
    print(json.dumps(dct, indent=2))
    
def wrapper(endpoint, data={}, base_url=base_url):
    try:
        response = get(endpoint, data, base_url)
        jprint(response)
    except requests.HTTPError as e:

        print(e)
        if e.response.status_code == 400:
            jprint(e.response.json())
        raise
    try:
        ctr = 1
        while True:
            query_status = get('%s/status/%s'% (endpoint.split('/')[0],response['request_id'],))
            jprint(query_status)
            if query_status['status'] !='running':
                # query has finished
                break
            else:
                time.sleep(ctr)
                ctr += 1
                #linear backoff
                print("Checking again")
    except requests.HTTPError as e:
        print(e)
        if e.response.status_code == 400:
            jprint(e.response.json())
        raise
    return pandas.concat(map(pandas.read_csv, query_status['request_uri']))
