import json
import uuid
from typing import Dict

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth


class AWSSearchDB:

    def __init__(self):
        host = 'search-tengine2-5nn4xi272rmoglndi7swodwzpy.us-east-1.es.amazonaws.com'
        port = 443
        # auth = ('tengine', '^T3ngine')
        credentials = boto3.Session(
            aws_access_key_id='AKIAUCYMDOCRXMRNIOPV',
            aws_secret_access_key='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',
        ).get_credentials()
        auth = AWSV4SignerAuth(credentials, 'us-east-1')

        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

    def export_anamnesis_to_file(self, filename):
        query = {
            'size': 1000
        }

        response = self.client.search(
            body=query,
            index='bapdb-reports-prod-6'
        )

        file1 = open(filename, "w")
        file1.write(str(response))
        file1.close()
