#!/bin/bash

# Replace this with your actual command
CMD="python speed_comp.py"

while true; do
    start=$(date +%s.%N)
    bash -c "$CMD"
    status=$?
    end=$(date +%s.%N)

    duration=$(echo "$end - $start" | bc)

    echo "Command took $duration seconds and exited with status $status"

    # Exit if it took >1s and exited normally (status 0)
    if (( $(echo "$duration > 1.0" | bc -l) )) && [ $status -eq 0 ]; then
        echo "Command ran too long and exited normally — exiting script."
        exit 0
    fi
done
