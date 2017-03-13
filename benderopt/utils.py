import json
import os


def get_test_data():
    return json.load(open("{}/tests/test_data.json".format(
        os.path.dirname(os.path.abspath(__file__))), "r"))
