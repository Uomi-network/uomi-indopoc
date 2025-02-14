import sys
import time
import os
import redis

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

# Functions
############################################

def snapshot_db(db, snapshot_key):
  snapshot_path = f'{SNAPSHOT_FOLDER}/{snapshot_key}/db_{db}'
  os.makedirs(snapshot_path, exist_ok=True)
  try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db, password=REDIS_PASS)
  except:
    print(f'❌ Error connecting to Redis DB: {db}')
    return
  # Store single datas as files
  for key in r.keys():
    with open(f'{snapshot_path}/{key}', 'w') as f:
      f.write(r.get(key).decode('utf-8'))

def snapshot():
  snapshot_key = int(time.time())
  print(f'🤖 Snapshot key: {snapshot_key}')

  for db in [REDIS_NODES_DB, REDIS_CHECKS_DB, REDIS_COMPLETITION_DB, REDIS_PROMPTS_DB]:
    print(f'- DB: {db}')
    snapshot_db(db, snapshot_key)
  for node in NODES:
    print(f'- Node: {node}')
    snapshot_db(node, snapshot_key)

  print(f'✅ Snapshot key: {snapshot_key} - Done!')

# Main
############################################

if __name__ == '__main__':
  while True:
    print('🚀 Taking snapshot...')
    snapshot()
    time.sleep(30)

