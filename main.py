import sys
import time
import os
import redis
import base64
import threading
import json
import traceback

import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurations
############################################

SIMULATION_MODE = False

NODE_ID = os.getenv('NODE_ID', 1)
PROMPTS_FILE_PATH = './prompts.txt'

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PASS = os.getenv('REDIS_PASS', '')
REDIS_PORT = 6379
REDIS_NODES_DB = 0
REDIS_CHECKS_DB = 13
REDIS_COMPLETITION_DB = 14
REDIS_PROMPTS_DB = 15
REDIS_NODE_INFERENCES_DB = NODE_ID

SEED = 42
MAX_NEW_TOKENS = 400
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K_DISPLAY = 5

NODES = [
  1, # RTX 4090
  2, # RTX A6000
  3, # H100 XMS
  4, # L40S
  5, # A100 SXM
  6, # LOCAL
]

# Redis Connections
############################################

r_nodes_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_NODES_DB, password=REDIS_PASS)
r_checks_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_CHECKS_DB, password=REDIS_PASS)
r_completition_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_COMPLETITION_DB, password=REDIS_PASS)
r_prompts_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_PROMPTS_DB, password=REDIS_PASS)
r_node_inferences_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_NODE_INFERENCES_DB, password=REDIS_PASS)

# Setup model
############################################

# - Set reproducibility settings
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# - Some additional flags to help reproducibility in certain cases:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# - Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# - Set the model and tokenizer
if not SIMULATION_MODE:
  model_name = "casperhansen/mistral-small-24b-instruct-2501-awq"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name)
  model.to(device)
  model.eval()  # put model in eval mode (no dropout, etc.)

# Functions
############################################

# This function generate a unique hash for a given string
def hash_string(input_string):
  input_bytes = input_string.encode('utf-8')
  return base64.b64encode(input_bytes).decode('utf-8')

# This function execute the inference and return the result
def execute_inference(prompt, key):
  if SIMULATION_MODE:
    time.sleep(2)
    return json.dumps({
      "key": key,
      "output": prompt,
      "execution_data": [],
      "executed_by": NODE_ID,
      "executed_in": 2
    })

  time_start = time.time()
  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

  execution_data = []

  for step in range(MAX_NEW_TOKENS):
    # Forward pass to get raw logits
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # Apply temperature (if not 1.0)
    if TEMPERATURE != 1.0:
      next_token_logits = next_token_logits / TEMPERATURE
    
    # Optional top-p filtering (here, top_p=1.0 => no filtering)
    if TOP_P < 1.0:
      sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
      sorted_logits_1d = sorted_logits[0]
      sorted_indices_1d = sorted_indices[0]

      sorted_probs = F.softmax(sorted_logits_1d, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

      cutoff_idx = torch.searchsorted(cumulative_probs, TOP_P)
      cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size- 1)
      sorted_logits_1d[cutoff_idx + 1:] = float('-inf')

      # Scatter back
      next_token_logits = torch.full_like(next_token_logits, float('-inf'))
      next_token_logits[0].scatter_(0, sorted_indices_1d, sorted_logits_1d)

    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]

    # Print the top-k tokens by probability
    top_probs, top_indices = probs.topk(TOP_K_DISPLAY, dim=-1)
    execution_data_top_k = []
    for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
      token_str = tokenizer.decode([idx.item()])
      execution_data_top_k.append({
        "str": token_str,
        "prob": prob.item(),
        "id": idx.item()
      })
      # print(f"   {rank}. '{token_str}' -> prob={prob.item():.6f}")

    # GREEDY selection instead of sampling
    # This ensures full determinism.
    next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
    selected_token_id = next_token_id.item()
    selected_token_str = tokenizer.decode([selected_token_id])
    selected_token_prob = probs[0, selected_token_id].item()

    # Append the chosen token
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Append the execution data
    execution_data.append({
      "str": selected_token_str,
      "prob": selected_token_prob,
      "id": selected_token_id,
      "top_k": execution_data_top_k
    })

  output = tokenizer.decode(input_ids[0], skip_special_tokens=True)

  result = {
    "key": key,
    "output": output,
    "execution_data": execution_data,
    "executed_by": NODE_ID,
    "executed_in": time.time() - time_start
  }
  return json.dumps(result)

# This function execute the check of an inference and return the result
# NOTE: Checking an inference means to take the output of the inference and check for each token if its probability is in the TOP_K_DISPLAY of the new inference
def execute_check(inference):
  if SIMULATION_MODE:
    time.sleep(2)
    return json.dumps({
      "key": "key",
      "check_result": True,
      "check_data": [],
      "checked_by": NODE_ID,
      "checked_in": 2,
      "executed_by": NODE_ID,
      "executed_in": 2
    })

  time_start = time.time()

  inference = json.loads(inference)
  prompt = r_prompts_db.get(inference["key"]).decode('utf-8')

  input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
  check_data = []
  check_result = True

  # Tokenize inference output
  inference_output_without_prompt = inference["output"][len(prompt):]
  inference_output_tokens = tokenizer.tokenize(inference_output_without_prompt)
  inference_output = tokenizer.convert_tokens_to_ids(inference_output_tokens)

  for step in range(MAX_NEW_TOKENS):
    # Forward pass to get raw logits
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]

    # Apply temperature (if not 1.0)
    if TEMPERATURE != 1.0:
      next_token_logits = next_token_logits / TEMPERATURE
    
    # Optional top-p filtering (here, top_p=1.0 => no filtering)
    if TOP_P < 1.0:
      sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
      sorted_logits_1d = sorted_logits[0]
      sorted_indices_1d = sorted_indices[0]

      sorted_probs = F.softmax(sorted_logits_1d, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

      cutoff_idx = torch.searchsorted(cumulative_probs, TOP_P)
      cutoff_idx = cutoff_idx.clamp(max=sorted_probs.size(-1) - 1)
      sorted_logits_1d[cutoff_idx + 1:] = float('-inf')

      # Scatter back
      next_token_logits = torch.full_like(next_token_logits, float('-inf'))
      next_token_logits[0].scatter_(0, sorted_indices_1d, sorted_logits_1d)

    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)  # shape: [1, vocab_size]

    # Take the current token from the inference output and check it's probability on the model
    check_data_top_k = []
    current_token_prob = None
    current_token_id = inference_output[step]
    current_token_str = tokenizer.decode([current_token_id])
    top_probs, top_indices = probs.topk(TOP_K_DISPLAY, dim=-1)
    for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
      token_str = tokenizer.decode([idx.item()])
      check_data_top_k.append({
        "str": token_str,
        "prob": prob.item(),
        "id": idx.item()
      })
      # print(f"   {rank}. '{token_str}' -> prob={prob.item():.6f}")
      if idx == current_token_id:
        current_token_prob = float(prob.item())
        break
    if current_token_prob is None:
      check_result = False
      print(f"❌ Current token: '{current_token_str}' -> not found in top-{TOP_K_DISPLAY}")
      break
    print(f"🤖 Current token: '{current_token_str}' -> prob={current_token_prob:.6f}")

    # GREEDY selection instead of sampling
    # This ensures full determinism.
    selected_token_str = tokenizer.decode([current_token_id])
    selected_token_prob = probs[0, current_token_id].item()

    # Append the chosen token
    next_token_id = torch.tensor([[current_token_id]]).to(device)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Append the check result
    check_data.append({
      "str": current_token_str,
      "prob": current_token_prob,
      "id": current_token_id,
      "top_k": check_data_top_k
    })

  result = {
    "key": inference["key"],
    "check_result": check_result,
    "check_data": check_data,
    "checked_by": NODE_ID,
    "checked_in": time.time() - time_start,
    "executed_by": inference["executed_by"],
    "executed_in": inference["executed_in"]
  }
  return json.dumps(result)

def loop_run():
  print("🧠 Node " + str(NODE_ID) + " is looping run...")
  
  try:
    remaining = 0

    # Start execution of the inferences (from r_prompts_db) and store the result in the node's db
    # NOTE: Ignore execution if it is already stored in the node's db
    prompts_runned_one = False
    r_prompts_db_keys = r_prompts_db.keys()
    r_prompts_db_keys = [key.decode('utf-8') for key in r_prompts_db_keys]
    for key in r_prompts_db_keys:
      prompt = r_prompts_db.get(key).decode('utf-8')
      if r_node_inferences_db.exists(key):
        print("Skipping inference: " + str(key))
      elif not prompts_runned_one:
        print("Executing inference: " + str(key))
        result = execute_inference(prompt, key)
        r_node_inferences_db.set(key, result)
        prompts_runned_one = True
      else:
        remaining += 1

    # Take list of other nodes from the r_nodes_db
    # nodes = r_nodes_db.keys()
    # nodes = [int(node) for node in nodes]
    nodes = [node for node in NODES if node != int(NODE_ID)]

    # Loop through the nodes, for each node take its inferences and execute the check
    check_runned_one = False
    for node in nodes:
      print("Checking node: " + str(node))
      # Try to connect to the node's db, if db not exists, skip the node
      try:
        node_inferences_db = r_node_inferences_db if node == NODE_ID else redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=node, password=REDIS_PASS)
      except:
        print("Skipping node: " + str(node))
        continue
      node_inferences_db_keys = node_inferences_db.keys()
      for key in node_inferences_db_keys:
        check_key = str(NODE_ID) + "_" + str(node) + "_" + str(key)
        if r_checks_db.exists(check_key):
          print("Skipping check: " + str(check_key))
        elif not check_runned_one:
          print("Executing check: " + str(check_key))
          inference = node_inferences_db.get(key).decode('utf-8')
          check_result = execute_check(inference)
          r_checks_db.set(check_key, check_result)
          check_runned_one = True
        else:
          remaining += 1

    # Store the node's db in the completition db
    r_completition_db.set(str(NODE_ID), remaining)
    print("✅ Node " + str(NODE_ID) + " completed the run loop.")
  except Exception as e:
    print("❌ Node " + str(NODE_ID) + " failed to complete the run loop.")
    print(traceback.format_exc())
    r_completition_db.set(str(NODE_ID), -1)

  # Re-run the loop
  time.sleep(1)
  loop_run()

# Ping
def loop_ping():
  print("💬 Node " + str(NODE_ID) + " is looping ping...")
  
  # Write the node id in the nodes db
  r_nodes_db.set(str(NODE_ID), 1, ex=30)

  # Re-run the loop
  time.sleep(10)
  loop_ping()

# Setup
def setup():
  # Read the prompts from the file (one inference per line)
  with open(PROMPTS_FILE_PATH, 'r') as f:
    prompts = f.readlines()
  # Normalize prompts by remove last character if is a new line
  prompts = [inference.rstrip('\n') for inference in prompts]
  # Store every inference in the redis db using the hash_string of the inference as key (if not already stored)
  for prompt in prompts:
    prompt_hash = hash_string(prompt)
    if not r_prompts_db.exists(prompt_hash):
      r_prompts_db.set(prompt_hash, prompt)

# Main
############################################

if __name__ == '__main__':
  # Setup
  setup()

  # Start the two loops threads and kill them if the main thread is killed
  loop_run_thread = threading.Thread(target=loop_run)
  loop_ping_thread = threading.Thread(target=loop_ping)
  loop_run_thread.start()
  loop_ping_thread.start()

  print("🚀 Node " + str(NODE_ID) + " is running...")
