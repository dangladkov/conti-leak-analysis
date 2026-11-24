#!/bin/bash

export OLLAMA_NUM_PARALLEL=6          # Total parallel threads
export OLLAMA_SCHED_SPREAD=1           # Spread across GPUs
export OLLAMA_KEEP_ALIVE="1h"          # Keep model in memory

ollama serve
