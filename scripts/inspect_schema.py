import json

def inspect():
    with open('data/raw/action-schema.json', 'r', encoding='utf-8') as f:
        act = json.load(f)
    print("=== ACTION SCHEMA ===")
    print("Title:", act.get("title"))
    print("Description:", act.get("description"))
    defs = act.get("$defs", {})
    print("Definitions in Action Schema:", list(defs.keys()))
    if "Actor" in defs:
        print("Actor fields:", json.dumps(defs["Actor"], indent=2))
    if "Repository" in defs:
        print("Repository fields:", json.dumps(defs["Repository"], indent=2))
    if "ActionDetails" in defs:
        print("ActionDetails keys:", list(defs["ActionDetails"].get("properties", {}).keys()))
        print("ActionDetails snippet:", json.dumps(defs["ActionDetails"], indent=2)[:1000])

    with open('data/raw/activity-schema.json', 'r', encoding='utf-8') as f:
        acty = json.load(f)
    print("\n=== ACTIVITY SCHEMA ===")
    print("Title:", acty.get("title"))
    print("Description:", acty.get("description"))
    defs_y = acty.get("$defs", {})
    print("Definitions in Activity Schema:", list(defs_y.keys()))
    if "Actor" in defs_y:
        print("Activity Actor:", json.dumps(defs_y["Actor"], indent=2))
    if "Activity" in defs_y:
        print("Activity enum/type:", json.dumps(defs_y["Activity"], indent=2))
    elif "properties" in acty and "activity" in acty["properties"]:
        print("Activity field:", json.dumps(acty["properties"]["activity"], indent=2))

if __name__ == "__main__":
    inspect()

