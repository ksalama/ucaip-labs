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
"""Utilities for generating BigQuery data querying scirpts."""

try:
    from utils.ucaip_utils import AIPUtils
except:
    from ucaip_utils import AIPUtils

def get_source_query(
    project, region, dataset_display_name, data_split, limit=None):
    
    aip_utils = AIPUtils(project, region)
    
    dataset = aip_utils.get_dataset_by_display_name(dataset_display_name)
    bq_source_uri = dataset.metadata['inputConfig']['bigquerySource']['uri']
    _, bq_dataset_name, bq_table_name = bq_source_uri.replace("g://", "").split('.')
    
    query = f'''
        SELECT 
            CAST(trip_start_timestamp AS STRING) trip_start_timestamp,
            IF(trip_month IS NULL, -1, trip_month) trip_month,	
            IF(trip_day IS NULL, -1, trip_day) trip_day,
            IF(trip_day_of_week IS NULL, -1, trip_day_of_week) trip_day_of_week,
            IF(trip_hour IS NULL, -1, trip_hour) trip_hour,	
            IF(trip_seconds IS NULL, -1, trip_seconds) trip_seconds,
            IF(trip_miles IS NULL, -1, trip_miles) trip_miles,
            IF(payment_type IS NULL, 'NA', payment_type) payment_type,
            IF(pickup_grid IS NULL, 'NA', pickup_grid) pickup_grid,
            IF(dropoff_grid IS NULL, 'NA', dropoff_grid) dropoff_grid,
            IF(euclidean IS NULL, -1, euclidean) euclidean,
            IF(loc_cross IS NULL, 'NA', loc_cross) loc_cross,
            tip_bin
        FROM {bq_dataset_name}.{bq_table_name} 
        WHERE data_split = '{data_split}'
    '''
    if limit:
        query += f'\n limit {limit}'
    return query