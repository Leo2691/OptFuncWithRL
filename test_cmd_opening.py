import subprocess

# Replace '/path/to/venv' with the path to your virtual environment
# Replace 'script.py' with the name of your Python script
# Replace 'arg1' and 'arg2' with your script's parameters respectively
process = subprocess.Popen(['/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal', '-e',
                                 '/venv/bin/python3', 'Hello_World.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

# Wait for the process to finish and get the output
stdout, stderr = process.communicate()

# Print the output
print(stdout.decode())