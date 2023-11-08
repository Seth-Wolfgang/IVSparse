#!/bin/bash


#
#   THIS FILE IS USED TO DUPLICATE RESULTS OF THE PAPER.
#   DEFAULT PARAMETERS USED IN PAPER WILL BE IN THE INITAL REPO CLONE
#   SEE simulatedBench.cpp FOR MORE INFORMATION ON PARAMETERS AND INSTRUCTIONS
#   ON HOW TO PLOT THE GENERATED DATA
#
#

file="simulatedBench.cpp"

# Function to replace density value in the file
replace_density() {
  local line_number="$1"
  local new_value="$2"
  
  # Get the current value of DENSITY
  old_value=$(sed -n "${line_number}p" "$file" | awk '{print $3}')

  # Escape old_value and new_value for use in sed
  old_value_escaped=$(printf '%s\n' "$old_value" | sed -e 's/[\/&]/\\&/g')
  new_value_escaped=$(printf '%s\n' "$new_value" | sed -e 's/[\/&]/\\&/g')

  # Replace the line
  sed -i "${line_number}s/#define DENSITY ${old_value_escaped}/#define DENSITY ${new_value_escaped}/" "$file"

  # Check if the line was replaced successfully
  if grep -q "#define DENSITY ${new_value_escaped}" "$file"; then
    g++ -O2 -w -I ~/eigen "$file" -o "${new_value}density"
    if [ -x "${new_value}density" ]; then
      ./"${new_value}density"
      rm "${new_value}density"
    else
      echo "Executable ${new_value}density was not created successfully."
    fi
  else
    echo "The line was not replaced successfully."
  fi
}

# Get the line number of the line with #define DENSITY
line_number=$(grep -n "#define DENSITY" "$file" | cut -d: -f1)

# Call the function for different new values
replace_density "$line_number" 0.01
replace_density "$line_number" 0.05
replace_density "$line_number" 0.1