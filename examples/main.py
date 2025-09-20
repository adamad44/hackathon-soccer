from database import *

# Get all matches from database as nested arrays
try:
    conn = get_connection('soccer_stats')
    cur = conn.cursor()
    cur.execute("SELECT * FROM epl_matches")
    matches = cur.fetchall()
    cur.close()
    conn.close()
    
    # Convert to nested arrays
    all_matches = []
    for match in matches:
        all_matches.append(list(match))
        
    print(f"Loaded {len(all_matches)} matches from database")
    
except Exception as e:
    print(f"Error loading matches: {e}")
    all_matches = []
    
print(all_matches[0])
    