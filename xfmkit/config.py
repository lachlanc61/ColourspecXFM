import os
import configparser
import json

config = configparser.ConfigParser()
config.read("conf/xfmkit.conf")

def get(section, value, default=None, mandatory=False):
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
        

def get_list(section, value, default=None, mandatory=False):
    """
    Reads config value from within section
    or return default
    """

    try:
        config_return = config.get(section, value)
        
        return json.loads(config_return)
    
    except:
        if mandatory: 
            raise Exception(f"Mandatory value {value} not found in section {section}")
        else:
            return default