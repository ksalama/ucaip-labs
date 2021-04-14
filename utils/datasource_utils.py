from google.cloud.aiplatform import gapic as aip

def get_source_query(
    project, region, dataset_display_name, data_split, limit=None):
    
    parent = f"projects/{project}/locations/{region}"
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    
    dataset_client = aip.DatasetServiceClient(client_options=client_options)
    for dataset in dataset_client.list_datasets(parent=parent):
        if dataset.display_name == dataset_display_name:
            dataset_uri = dataset.name
            break

    dataset = dataset_client.get_dataset(name=dataset_uri)
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