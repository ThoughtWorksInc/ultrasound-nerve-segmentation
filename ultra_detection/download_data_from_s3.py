import os

import boto3


def download_data(path, limit=None):
  if not os.path.exists(path):
    os.makedirs(path)

  bucket = boto3.resource('s3').Bucket('thoughtworks-ultrasound')

  objects = bucket.objects.limit(limit) if limit else bucket.objects.all()
  for key in objects:
    if key.key.endswith('/'):
      os.makedirs(os.path.join(path, key.key))
    else:
      key.Object().download_file(os.path.join(path, key.key))
