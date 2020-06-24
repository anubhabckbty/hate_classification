### Requirements:
* Python 3 (version 3.7.6 or below)
(note this program does not run in Python 3.8)
* **data** folder containing the **tweets.csv** file

### Process:
* Run './setupEnv.sh virtualenv' create environment with appropriate package versions
* Move into the environment by running, 'source virtualenv/bin/activate'
* When inside the environment run, 'python access.py'

Whenever the program is needed to be run, changing environment is necessary
You can move out of the environment by using the 'deactivate' command

### Output:
The default output file is data/classified_tweets.csv

### Configuration:
to change input or output, or any other configuration, the code inside classify.py needs to be changed under the configuration section.
The chunksize variable takes in the number of rows to take at a time

