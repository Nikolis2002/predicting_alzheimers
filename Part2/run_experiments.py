import pymongo
import subprocess

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
col = client['genetic_algos']['resultsv2']

# Fetch existing tags from the database
existing_tags = set(
    doc['params']['tag']
    for doc in col.find({}, {'params.tag': 1})
    if 'params' in doc and 'tag' in doc['params']
)

# Experiment configurations
pops =     [20,  20,  20,  20,  20,  200, 200, 200, 200, 200]
pcs =   [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
mus =  [0.00,0.01,0.10,0.01,0.01,0.00,0.01,0.10,0.01,0.01]
ELITE_RATE = 5

# Iterate through all expected tags and run missing ones
for row in range(1, 11):
    pop = pops[row-1]
    pc  = pcs[row-1]
    mu  = mus[row-1]
    elitism = (pop * ELITE_RATE) // 100
    for run in range(1, 11):
        tag = f"row{row}_run{run}"
        if tag in existing_tags:
            continue

        print(f"Running missing: {tag} (pop={pop}, pc={pc}, mu={mu}, elitism={elitism})")
        subprocess.run([
            "python", "genetic_algo.py",
            "--N", str(pop),
            "--pc", str(pc),
            "--mu", str(mu),
            "--elitism", str(elitism),
            "--sel", "tournament",
            "--cross", "uniform",
            "--adaptive_elitism", "false",
            "--injection", "false",
            "--tag", tag
        ])

print("Done. All missing runs have been executed.")