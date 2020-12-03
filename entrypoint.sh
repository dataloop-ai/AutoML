#!/bin/sh

# Start the ssh server
/etc/init.d/ssh restart

# Execute the CMD
exec "$@"