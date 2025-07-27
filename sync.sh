#!/bin/bash

while true; do
  rsync -avz --delete \
    --exclude 'datasets/' \
    --exclude 'assets/' \
    --exclude 'checkpoints/' \
    --exclude 'logs/' \
    --exclude 'test_data/' \
    -e ssh ./ \
    hp@hp.neural-love.com:/home/hp/projects/artem/chessml_sync
  
  sleep 1
done