import json

with open('data/raw/activity-schema.json', 'r', encoding='utf-8') as f:
    acty = json.load(f)

print("Activity Properties:")
print(json.dumps(acty['properties'], indent=2)[:1500])

