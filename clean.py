import sys
import time
import os
import redis
import json

# Configurations
############################################

SNAPSHOT_FOLDER = os.getenv('SNAPSHOT_FOLDER', './snapshots')

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASS = os.getenv('REDIS_PASS', '')
REDIS_PORT = 6379
REDIS_NODES_DB = 0
REDIS_CHECKS_DB = 13
REDIS_COMPLETITION_DB = 14
REDIS_PROMPTS_DB = 15

NODES = [
  1, # RTX 4090
  2, # RTX A6000
  3, # H100 XMS
  4, # L40S
  5, # A100 SXM
  6, # LOCAL
]

CLEAN_KEY = 'e9b08d3217a23e0b020fab8f4ab7ac4ed831f8aa7f0f60daaf3dc1bf00935833'

# Functions
############################################

def clean_db(db):
  r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db, password=REDIS_PASS)
  keys = r.keys()
  print(f'  - Keys: {len(keys)}')

  for key in keys:
    key = key.decode('utf-8')
    if CLEAN_KEY in key:
      print(f'    - Deleting: {key}')
      r.delete(key)

# Connect to every db, if a key include CLEAN_KEY, delete it
def clean():
  for db in [REDIS_NODES_DB, REDIS_CHECKS_DB, REDIS_COMPLETITION_DB, REDIS_PROMPTS_DB]:
    print(f'- DB: {db}')
    clean_db(db)
  for node in NODES:
    print(f'- Node: {node}')
    clean_db(node)

# Main
############################################

if __name__ == '__main__':
  if CLEAN_KEY == '':
    print('❌ Please set CLEAN_KEY')
    sys.exit(1)
  
  clean()
