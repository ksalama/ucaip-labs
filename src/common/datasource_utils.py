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


from google.cloud import aiplatform as vertex_ai

def get_bq_uri(dataset_display_name):
    datasets = vertex_ai.TabularDataset.list(filter=f"display_name={dataset_display_name}")
    dataset = datasets[0]
    bq_source_uri = dataset.gca_resource.metadata["inputConfig"]["bigquerySource"][
        "uri"
    ]
    _, bq_dataset_name, bq_table_name = bq_source_uri.replace("g://", "").split(".")
    return bq_source_uri, bq_dataset_name, bq_table_name

def create_bq_source_query(dataset_display_name, missing, label_column=None, ML_use=None, limit=None):
    _, bq_dataset_name, bq_table_name = get_bq_uri(dataset_display_name)

    query = """
    SELECT
"""
    for column, transform in missing.items():
        if isinstance(transform, str):
            query += f"""        IF({column} IS NULL, '{transform}', {column}) {column},
"""
        else:
            query += f"""        IF({column} IS NULL, {transform}, {column}) {column},
"""

    if label_column:
        query += f"""        {label_column}
"""

    query += f"""    FROM {bq_dataset_name}.{bq_table_name}
"""

    if ML_use:
        query += f"""    WHERE ML_use = '{ML_use}'
"""

    if limit:
        query += f"""    LIMIT {limit}"""

    return query

