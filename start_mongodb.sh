#!/usr/bin/env bash

db_path=$(cat config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['logging']['mongodb_path'])")
db_port=$(cat config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['logging']['port'])")
mongod --dbpath $db_path --port $db_port