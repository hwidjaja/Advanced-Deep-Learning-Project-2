import os
import json


def create_json(target_dir, filename, dict_to_save={}):
    json_as_string = json.dumps(dict_to_save, indent=4, sort_keys=False)
    
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    
    with open(os.path.join(target_dir, filename), "w") as outfile:
        outfile.write(json_as_string)
            

def update_json(target_dir, filename, dict_to_save={}):
    
    f = open(os.path.join(target_dir, filename), "r")
    old_dict = json.loads(f.read())
    old_dict.update(dict_to_save)

    json_as_string = json.dumps(old_dict, indent=4, sort_keys=False)
    with open(os.path.join(target_dir, filename), "w") as outfile:
        outfile.write(json_as_string)
    
    f.close()
    

def load_json(target_dir, filename):
    f = open(os.path.join(target_dir, filename), "r")
    return json.load(f)