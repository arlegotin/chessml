#!/bin/bash

REMOTE="hp@hp.neural-love.com"
REMOTE_PATH="/home/hp/projects/artem/chessml_sync/test_data/"
LOCAL_PATH="./test_data/"

# Create local test_data dir if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Sync only missing files from remote to local
rsync -avz --ignore-existing -e ssh "$REMOTE:$REMOTE_PATH" "$LOCAL_PATH"