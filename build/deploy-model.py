# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create endpoint and deploy a model to AI Platform."""


import argparse
import os
import sys
import logging
import json


SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.utils.ucaip_utils import AIPUtils

SERVING_SPEC_FILEPATH = 'build/serving_resources_spec.json'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        type=str,
    )

    parser.add_argument(
        '--project',  
        type=str,
    )
    
    parser.add_argument(
        '--region',  
        type=str,
    )
    
    parser.add_argument(
        '--endpoint-display-name', 
        type=str,
    )

    parser.add_argument(
        '--model-display-name', 
        type=str,
    )

    return parser.parse_args()


def create_endpoint(project, region, endpoint_display_name):
    logging.info(f"Creating endpoint {endpoint_display_name}")
    aip_utils = AIPUtils(project, region)
    result = aip_utils.create_endpoint(endpoint_display_name).result()
    logging.info(f"Endpoint is ready.")
    return result
    
def deploy_model(project, region, endpoint_display_name, model_display_name, dedicated_serving_resources_spec):
    logging.info(f"Deploying model {model_display_name} to  endpoint {endpoint_display_name}")
    aip_utils = AIPUtils(project, region)
    result = aip_utils.deploy_model(
        model_display_name,
        endpoint_display_name,
        dedicated_serving_resources_spec).result()
    logging.info(f"Model {model_display_name} is deployed.")
    return result
    

def main():
    args = get_args()
    
    if args.mode == 'endpoint':
        result = create_endpoint(
            args.project, 
            args.region, 
            args.endpoint_display_name
        )
    else:
        if not args.model_display_name:
            raise ValueError("model-display-name must be supplied.")
        with open(SERVING_SPEC_FILEPATH) as json_file:
            dedicated_serving_resources_spec = json.load(json_file)
        logging.info(f"serving resources: {dedicated_serving_resources_spec}")
        result = deploy_model(
            args.project, 
            args.region, 
            args.endpoint_display_name, 
            args.model_display_name,
            dedicated_serving_resources_spec
        )
    logging.info(result)
        
    
if __name__ == "__main__":
    main()
    
