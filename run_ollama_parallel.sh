#!/bin/bash

export OLLAMA_NUM_PARALLEL=8          # Total parallel threads
export OLLAMA_SCHED_SPREAD=1          # Spread across GPUs
export OLLAMA_KEEP_ALIVE="1h"         # Keep model in memory
export OLLAMA_FLASH_ATTENTION=true    # Speed up attention layers

ollama serve
