import os
import sys
import configparser
import json
from pathlib import Path


CONF_FILE_DEFAULT="conf/xfmkit.conf"

config = configparser.ConfigParser()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__base__ = os.path.dirname(__location__)

def setup(conf_file=CONF_FILE_DEFAULT):
    """
    Initialise config file
    """
    conf_location = os.path.join(__base__,conf_file)

    try:
        with open(conf_location) as f:
            config.read_file(f)
            print(f"conf_loaded from {conf_location}")
    except IOError:
        raise FileNotFoundError(f"{conf_location} not found")

    return 

setup(conf_file=CONF_FILE_DEFAULT)

def get(section, value, default=None, mandatory=True):
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


def get_str(section, value, default=None, mandatory=True):
    """
    Reads config value as string (needs cast when called)
    or return default
    """

    try:
        return config.get(section, value).strip()
    
    except:
        if mandatory: 
            raise Exception(f"Mandatory value {value} not found in section {section}")
        else:
            return default
        