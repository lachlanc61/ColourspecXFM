import os
import sys
import configparser
import json
from pathlib import Path

CONF_FILE="conf/xfmkit.conf"

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__base__ = os.path.dirname(__location__)
conf_location = os.path.join(__base__,CONF_FILE)

config = configparser.ConfigParser()

try:
    with open(conf_location) as f:
        config.read_file(f)
        print(f"conf_loaded from {conf_location}")
except IOError:
    raise FileNotFoundError(f"{conf_location} not found")


def get_str(section, value, default=None, mandatory=False):
    """
    Reads config value from within section
    or return default
    """

    try:
        return config.get(section, value).strip()
    
    except:
        if mandatory: 
            raise Exception(f"Mandatory value {value} not found in section {section}")
        else:
            return default
        

def get(section, value, default=None, mandatory=False):
    """
    Read config value/list

    attempt cast via json.loads
    """

    try:
        config_return = config.get(section, value)
        
        return json.loads(config_return)
    
    except:
        if mandatory: 
            raise Exception(f"Mandatory value {value} not found in section {section}")
        else:
            return default