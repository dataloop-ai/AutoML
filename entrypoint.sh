#!/bin/sh

# Start the ssh server
service ssh start

# Execute the CMD
exec "$@"