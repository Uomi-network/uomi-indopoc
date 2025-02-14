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

def snapshot_db(db, override=False):
  snapshot_path = f'{SNAPSHOT_FOLDER}/db_{db}'
  os.makedirs(snapshot_path, exist_ok=True)
  try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db, password=REDIS_PASS)
  except:
    print(f'❌ Error connecting to Redis DB: {db}')
    return
  # Store single datas as files
  keys = r.keys()
  print(f'  - Keys: {len(keys)}')
  for key in keys:
    key = key.decode('utf-8')
    # Skip key if already stored
    if not override and os.path.exists(f'{snapshot_path}/{key}'):
      continue
    with open(f'{snapshot_path}/{key}', 'w') as f:
      f.write(r.get(key).decode('utf-8'))

def snapshot():
  snapshot_key = int(time.time())
  print(f'🤖 Snapshot start')

  for db in [REDIS_NODES_DB, REDIS_CHECKS_DB, REDIS_COMPLETITION_DB, REDIS_PROMPTS_DB]:
    print(f'- DB: {db}')
    snapshot_db(db, db != REDIS_PROMPTS_DB)
  for node in NODES:
    print(f'- Node: {node}')
    snapshot_db(node)

  print(f'✅ Snapshot done!')

def recap():
  print(f'📊 Recap')
  # Calculate total prompts count (number of keys in the db_15 snapshot)
  total_prompts = 0
  for key in os.listdir(f'{SNAPSHOT_FOLDER}/db_15'):
    total_prompts += 1
  print(f'- Total prompts: {total_prompts}')
  # Calculate stats for each node
  for node in NODES:
    print(f'- Node: {node}')
    # Read remaining prompts from db_14 snapshot
    remaining_prompts = None
    if os.path.exists(f'{SNAPSHOT_FOLDER}/db_14/{node}'):
      with open(f'{SNAPSHOT_FOLDER}/db_14/{node}', 'r') as f:
        remaining_prompts = int(f.read())
    print(f'  - Remaining prompts: {remaining_prompts}/{total_prompts}')
  
  print(f'✅ Recap done!')

# Main
############################################

if __name__ == '__main__':
  while True:
    print(' ')
    snapshot()
    print(' ')
    recap()
    print('*'*50)
    time.sleep(30)

