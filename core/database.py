import mariadb
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection configuration from environment variables
DB_CONFIG = {
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306"))
}

def get_connection(database=None):
    """Get a connection to MariaDB"""
    if not DB_CONFIG["password"]:
        raise ValueError("DB_PASSWORD environment variable is not set. Check your .env file.")
    
    config = DB_CONFIG.copy()
    if database:
        config["database"] = database
    return mariadb.connect(**config)

def create_database(database_name):
    """Create a new database"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"Database '{database_name}' created successfully")
        cur.close()
        conn.close()
        return True
    except mariadb.Error as e:
        print(f"Error creating database: {e}")
        return False

def list_databases():
    """List all databases"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        databases = cur.fetchall()
        print("Available databases:")
        for db in databases:
            print(f"  - {db[0]}")
        cur.close()
        conn.close()
        return [db[0] for db in databases]
    except mariadb.Error as e:
        print(f"Error listing databases: {e}")
        return []

def execute_query(query, database=None):
    """Execute a SQL query"""
    try:
        conn = get_connection(database)
        cur = conn.cursor()
        cur.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            results = cur.fetchall()
            cur.close()
            conn.close()
            return results
        else:
            conn.commit()
            cur.close()
            conn.close()
            return True
    except mariadb.Error as e:
        print(f"Error executing query: {e}")
        return None

def get_table_columns(table_name, database='soccer_stats'):
    """Get detailed column information for a table"""
    query = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COLUMN_COMMENT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{database}' 
        AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
    """
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        columns = cur.fetchall()
        cur.close()
        conn.close()
        
        return [{'name': col[0], 'type': col[1], 'nullable': col[2], 
                'default': col[3], 'comment': col[4]} for col in columns]
    except mariadb.Error as e:
        print(f"Error getting table columns: {e}")
        return []

def show_table_structure(table_name='epl_matches', database='soccer_stats'):
    """Display the structure of a table in a formatted way"""
    columns = get_table_columns(table_name, database)
    
    if not columns:
        print(f"No columns found for table '{table_name}' in database '{database}'")
        return
    
    print(f"\n=== {table_name.upper()} TABLE STRUCTURE ===")
    print(f"Database: {database}")
    print(f"Columns: {len(columns)}")
    print("\n" + "="*80)
    print(f"{'Column Name':<25} | {'Data Type':<15} | {'Nullable':<8} | {'Default':<15}")
    print("="*80)
    
    for col in columns:
        default_val = str(col['default']) if col['default'] is not None else 'NULL'
        if len(default_val) > 14:
            default_val = default_val[:11] + '...'
        
        print(f"{col['name']:<25} | {col['type']:<15} | {col['nullable']:<8} | {default_val:<15}")
    
    print("="*80)
    
    return columns

def get_sample_data(table_name='epl_matches', database='soccer_stats', limit=3):
    """Get sample data from the table to see what the data looks like"""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    
    try:
        conn = get_connection(database)
        cur = conn.cursor()
        cur.execute(query)
        results = cur.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        
        return {'columns': columns, 'data': results}
    except mariadb.Error as e:
        print(f"Error getting sample data: {e}")
        return None

def show_sample_data(table_name='epl_matches', database='soccer_stats', limit=3):
    """Display sample data from the table"""
    sample = get_sample_data(table_name, database, limit)
    
    if not sample:
        print(f"No sample data available for table '{table_name}'")
        return
    
    print(f"\n=== SAMPLE DATA FROM {table_name.upper()} ===")
    print(f"Showing {len(sample['data'])} of {limit} requested rows")
    print("\n" + "-"*120)
    
    # Print column headers
    header_line = ""
    for col in sample['columns']:
        header_line += f"{col:<15} | "
    print(header_line.rstrip(" | "))
    print("-"*120)
    
    # Print data rows
    for row in sample['data']:
        data_line = ""
        for value in row:
            str_val = str(value) if value is not None else 'NULL'
            if len(str_val) > 14:
                str_val = str_val[:11] + '...'
            data_line += f"{str_val:<15} | "
        print(data_line.rstrip(" | "))
    
    print("-"*120)
    
    return sample

def get_column_names(table_name='epl_matches', database='soccer_stats'):
    """Get just the column names as a simple list"""
    columns = get_table_columns(table_name, database)
    return [col['name'] for col in columns]

def print_column_names(table_name='epl_matches', database='soccer_stats'):
    """Print column names in a simple list format"""
    columns = get_column_names(table_name, database)
    
    print(f"\nCOLUMN NAMES FOR {table_name.upper()}:")
    print("=" * 40)
    for i, col in enumerate(columns, 1):
        print(f"{i:2}. {col}")
    print(f"\nTotal: {len(columns)} columns")
    
    return columns

def setup_soccer_database():
    """Create soccer_stats database and EPL matches table"""
    
    # Create database
    if not create_database('soccer_stats'):
        return False
    
    # Create the EPL matches table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS epl_matches (
        id INT AUTO_INCREMENT PRIMARY KEY,
        season VARCHAR(10) NOT NULL,
        match_date DATE NOT NULL,
        home_team VARCHAR(50) NOT NULL,
        away_team VARCHAR(50) NOT NULL,
        full_time_home_goals INT NOT NULL,
        full_time_away_goals INT NOT NULL,
        full_time_result CHAR(1) NOT NULL,
        half_time_home_goals INT NOT NULL,
        half_time_away_goals INT NOT NULL,
        half_time_result CHAR(1) NOT NULL,
        home_shots INT NOT NULL,
        away_shots INT NOT NULL,
        home_shots_on_target INT NOT NULL,
        away_shots_on_target INT NOT NULL,
        home_corners INT NOT NULL,
        away_corners INT NOT NULL,
        home_fouls INT NOT NULL,
        away_fouls INT NOT NULL,
        home_yellow_cards INT NOT NULL,
        away_yellow_cards INT NOT NULL,
        home_red_cards INT NOT NULL,
        away_red_cards INT NOT NULL,
        INDEX idx_season (season),
        INDEX idx_date (match_date),
        INDEX idx_home_team (home_team),
        INDEX idx_away_team (away_team)
    ) ENGINE=InnoDB;
    """
    
    if execute_query(create_table_query, 'soccer_stats'):
        print("EPL matches table created successfully")
        return True
    else:
        print("Failed to create EPL matches table")
        return False

def restore_from_backup():
    """Restore database from SQL backup file (if needed)"""
    import subprocess
    import os
    
    if not os.path.exists('soccer_stats_backup.sql'):
        print("No backup file found. Database should already contain data.")
        return False
    
    try:
        # Restore from backup
        cmd = [
            "C:\\Program Files\\MariaDB 11.8\\bin\\mysql.exe",
            "-u", "root", f"-p{DB_CONFIG['password']}", 
            "soccer_stats"
        ]
        
        with open('soccer_stats_backup.sql', 'r') as backup_file:
            result = subprocess.run(cmd, stdin=backup_file, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Database restored from backup successfully!")
            return True
        else:
            print(f"Error restoring from backup: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error restoring from backup: {e}")
        return False

# ===== EXAMPLE USE CASES FOR LEARNING =====

def example_basic_queries():
    """Examples of basic SELECT queries"""
    print("\n=== BASIC QUERIES EXAMPLES ===")
    
    # 1. Get total number of matches
    result = execute_query("SELECT COUNT(*) as total_matches FROM epl_matches", 'soccer_stats')
    if result:
        print(f"1. Total matches in database: {result[0][0]}")
    
    # 2. Get matches for a specific team
    result = execute_query("SELECT season, match_date, home_team, away_team, full_time_home_goals, full_time_away_goals FROM epl_matches WHERE home_team = 'Arsenal' OR away_team = 'Arsenal' LIMIT 5", 'soccer_stats')
    if result:
        print("\n2. Arsenal's first 5 matches:")
        for match in result:
            season, date, home, away, home_goals, away_goals = match
            print(f"   {season}: {home} {home_goals}-{away_goals} {away} ({date})")
    
    # 3. High-scoring matches (6+ goals)
    result = execute_query("SELECT season, match_date, home_team, away_team, full_time_home_goals, full_time_away_goals FROM epl_matches WHERE (full_time_home_goals + full_time_away_goals) >= 6 ORDER BY (full_time_home_goals + full_time_away_goals) DESC LIMIT 5", 'soccer_stats')
    if result:
        print("\n3. Top 5 highest-scoring matches:")
        for match in result:
            season, date, home, away, home_goals, away_goals = match
            total = home_goals + away_goals
            print(f"   {home} {home_goals}-{away_goals} {away} ({total} goals total) - {season}")

def example_aggregation_queries():
    """Examples of GROUP BY and aggregation queries"""
    print("\n=== AGGREGATION QUERIES EXAMPLES ===")
    
    # 1. Goals per season
    result = execute_query("""
        SELECT season, 
               SUM(full_time_home_goals + full_time_away_goals) as total_goals,
               AVG(full_time_home_goals + full_time_away_goals) as avg_goals_per_match,
               COUNT(*) as matches
        FROM epl_matches 
        GROUP BY season 
        ORDER BY season DESC 
        LIMIT 5
    """, 'soccer_stats')
    
    if result:
        print("\n1. Goals statistics by season (last 5 seasons):")
        print("   Season    | Total Goals | Avg/Match | Matches")
        print("   ----------|-------------|-----------|--------")
        for row in result:
            season, total, avg, matches = row
            print(f"   {season:<9} | {total:<11} | {avg:<8.2f}  | {matches}")
    
    # 2. Team performance as home vs away
    result = execute_query("""
        SELECT 
            'Home' as venue,
            AVG(full_time_home_goals) as avg_goals_scored,
            AVG(full_time_away_goals) as avg_goals_conceded,
            SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total_matches
        FROM epl_matches
        UNION ALL
        SELECT 
            'Away' as venue,
            AVG(full_time_away_goals) as avg_goals_scored,
            AVG(full_time_home_goals) as avg_goals_conceded,
            SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total_matches
        FROM epl_matches
    """, 'soccer_stats')
    
    if result:
        print("\n2. Home vs Away performance across all matches:")
        print("   Venue | Avg Goals For | Avg Goals Against | Wins | Total Matches | Win %")
        print("   ------|---------------|-------------------|------|---------------|------")
        for row in result:
            venue, goals_for, goals_against, wins, total = row
            win_pct = (wins / total) * 100
            print(f"   {venue:<5} | {goals_for:<13.2f} | {goals_against:<17.2f} | {wins:<4} | {total:<13} | {win_pct:.1f}%")

def example_advanced_queries():
    """Examples of more complex queries with JOINs and subqueries"""
    print("\n=== ADVANCED QUERIES EXAMPLES ===")
    
    # 1. Team win percentages in recent seasons (simplified approach)
    result = execute_query("""
        SELECT 
            team,
            SUM(wins) as total_wins,
            SUM(matches) as total_matches,
            ROUND((SUM(wins) * 100.0 / SUM(matches)), 1) as win_percentage
        FROM (
            SELECT 
                home_team as team,
                SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as wins,
                COUNT(*) as matches
            FROM epl_matches 
            WHERE season >= '2020/21'
            GROUP BY home_team
            
            UNION ALL
            
            SELECT 
                away_team as team,
                SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as wins,
                COUNT(*) as matches
            FROM epl_matches 
            WHERE season >= '2020/21'
            GROUP BY away_team
        ) combined_stats
        GROUP BY team
        ORDER BY win_percentage DESC 
        LIMIT 8
    """, 'soccer_stats')
    
    if result:
        print("\n1. Team win percentages since 2020/21:")
        print("   Team           | Wins | Matches | Win %")
        print("   ---------------|------|---------|------")
        for row in result:
            team, wins, matches, win_pct = row
            print(f"   {team:<14} | {wins:<4} | {matches:<7} | {win_pct}%")
    
    # 2. Head-to-head record between two teams
    result = execute_query("""
        SELECT 
            CONCAT(home_team, ' vs ', away_team) as matchup,
            COUNT(*) as total_matches,
            SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as away_wins,
            SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
            AVG(full_time_home_goals + full_time_away_goals) as avg_goals
        FROM epl_matches 
        WHERE (home_team = 'Arsenal' AND away_team = 'Tottenham') 
           OR (home_team = 'Tottenham' AND away_team = 'Arsenal')
        GROUP BY CONCAT(home_team, ' vs ', away_team)
        ORDER BY home_team
    """, 'soccer_stats')
    
    if result:
        print("\n2. North London Derby (Arsenal vs Tottenham):")
        for row in result:
            matchup, total, home_wins, away_wins, draws, avg_goals = row
            print(f"   {matchup}: {total} matches")
            print(f"      Wins: {home_wins} | {away_wins} | {draws} draws")
            print(f"      Average goals per match: {avg_goals:.2f}")
    
    # 3. Monthly goal scoring trends
    result = execute_query("""
        SELECT 
            MONTH(match_date) as month,
            MONTHNAME(match_date) as month_name,
            COUNT(*) as matches,
            AVG(full_time_home_goals + full_time_away_goals) as avg_goals_per_match
        FROM epl_matches 
        WHERE season >= '2020/21'
        GROUP BY MONTH(match_date), MONTHNAME(match_date)
        ORDER BY month
    """, 'soccer_stats')
    
    if result:
        print("\n3. Goals per match by month (since 2020/21):")
        print("   Month     | Matches | Avg Goals")
        print("   ----------|---------|----------")
        for row in result:
            month_num, month_name, matches, avg_goals = row
            print(f"   {month_name:<9} | {matches:<7} | {avg_goals:.2f}")

# ===== NO-SQL DATA ACCESS FUNCTIONS =====
# All functions below let you get data without writing any SQL!

def get_all_teams():
    """Get list of all teams in the database"""
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT home_team FROM epl_matches ORDER BY home_team")
        teams = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return teams
    except Exception as e:
        print(f"Error getting teams: {e}")
        return []

def get_all_seasons():
    """Get list of all seasons in the database"""
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT season FROM epl_matches ORDER BY season")
        seasons = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return seasons
    except Exception as e:
        print(f"Error getting seasons: {e}")
        return []

def get_team_stats(team_name, season=None):
    """Get comprehensive stats for a team (optionally for specific season)"""
    season_filter = f"AND season = '{season}'" if season else ""
    
    query = f"""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN (home_team = '{team_name}' AND full_time_result = 'H') OR 
                         (away_team = '{team_name}' AND full_time_result = 'A') THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
            SUM(CASE WHEN (home_team = '{team_name}' AND full_time_result = 'A') OR 
                         (away_team = '{team_name}' AND full_time_result = 'H') THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN home_team = '{team_name}' THEN full_time_home_goals ELSE full_time_away_goals END) as goals_scored,
            SUM(CASE WHEN home_team = '{team_name}' THEN full_time_away_goals ELSE full_time_home_goals END) as goals_conceded,
            AVG(CASE WHEN home_team = '{team_name}' THEN full_time_home_goals ELSE full_time_away_goals END) as avg_goals_scored,
            AVG(CASE WHEN home_team = '{team_name}' THEN full_time_away_goals ELSE full_time_home_goals END) as avg_goals_conceded
        FROM epl_matches 
        WHERE (home_team = '{team_name}' OR away_team = '{team_name}') {season_filter}
    """
    
    result = execute_query(query, 'soccer_stats')
    if result:
        stats = result[0]
        return {
            'team': team_name,
            'season': season or 'All seasons',
            'matches_played': stats[0],
            'wins': stats[1],
            'draws': stats[2], 
            'losses': stats[3],
            'goals_scored': stats[4],
            'goals_conceded': stats[5],
            'goal_difference': stats[4] - stats[5],
            'avg_goals_scored': round(stats[6], 2),
            'avg_goals_conceded': round(stats[7], 2),
            'points': stats[1] * 3 + stats[2],
            'win_percentage': round((stats[1] / stats[0] * 100), 1) if stats[0] > 0 else 0
        }
    return None

def get_head_to_head(team1, team2, limit=None):
    """Get head-to-head record between two teams"""
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
        SELECT season, match_date, home_team, away_team, 
               full_time_home_goals, full_time_away_goals, full_time_result
        FROM epl_matches 
        WHERE (home_team = '{team1}' AND away_team = '{team2}') OR 
              (home_team = '{team2}' AND away_team = '{team1}')
        ORDER BY match_date DESC 
        {limit_clause}
    """
    
    matches = execute_query(query, 'soccer_stats')
    if not matches:
        return None
    
    team1_wins = sum(1 for m in matches if (m[2] == team1 and m[6] == 'H') or (m[3] == team1 and m[6] == 'A'))
    team2_wins = sum(1 for m in matches if (m[2] == team2 and m[6] == 'H') or (m[3] == team2 and m[6] == 'A'))
    draws = sum(1 for m in matches if m[6] == 'D')
    
    return {
        'team1': team1,
        'team2': team2,
        'total_matches': len(matches),
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'recent_matches': matches[:10] if len(matches) > 10 else matches
    }

def get_top_scorers(season=None, limit=10):
    """Get teams with most goals scored (optionally for specific season)"""
    season_filter = f"WHERE season = '{season}'" if season else ""
    
    query = f"""
        SELECT team, SUM(goals_scored) as total_goals, COUNT(*) as matches,
               ROUND(AVG(goals_scored), 2) as avg_goals_per_match
        FROM (
            SELECT home_team as team, full_time_home_goals as goals_scored, season
            FROM epl_matches {season_filter}
            UNION ALL
            SELECT away_team as team, full_time_away_goals as goals_scored, season
            FROM epl_matches {season_filter}
        ) all_goals
        GROUP BY team
        ORDER BY total_goals DESC
        LIMIT {limit}
    """
    
    result = execute_query(query, 'soccer_stats')
    return [{'team': r[0], 'total_goals': r[1], 'matches': r[2], 'avg_per_match': r[3]} for r in result] if result else []

def get_best_defenses(season=None, limit=10):
    """Get teams with fewest goals conceded (optionally for specific season)"""
    season_filter = f"WHERE season = '{season}'" if season else ""
    
    query = f"""
        SELECT team, SUM(goals_conceded) as total_conceded, COUNT(*) as matches,
               ROUND(AVG(goals_conceded), 2) as avg_conceded_per_match
        FROM (
            SELECT home_team as team, full_time_away_goals as goals_conceded, season
            FROM epl_matches {season_filter}
            UNION ALL
            SELECT away_team as team, full_time_home_goals as goals_conceded, season
            FROM epl_matches {season_filter}
        ) all_conceded
        GROUP BY team
        ORDER BY total_conceded ASC
        LIMIT {limit}
    """
    
    result = execute_query(query, 'soccer_stats')
    return [{'team': r[0], 'total_conceded': r[1], 'matches': r[2], 'avg_per_match': r[3]} for r in result] if result else []

def get_high_scoring_matches(min_goals=5, season=None, limit=20):
    """Get matches with high goal counts"""
    season_filter = f"AND season = '{season}'" if season else ""
    
    query = f"""
        SELECT season, match_date, home_team, away_team, 
               full_time_home_goals, full_time_away_goals,
               (full_time_home_goals + full_time_away_goals) as total_goals
        FROM epl_matches 
        WHERE (full_time_home_goals + full_time_away_goals) >= {min_goals} {season_filter}
        ORDER BY total_goals DESC, match_date DESC
        LIMIT {limit}
    """
    
    result = execute_query(query, 'soccer_stats')
    return [{'season': r[0], 'date': str(r[1]), 'home_team': r[2], 'away_team': r[3], 
             'home_goals': r[4], 'away_goals': r[5], 'total_goals': r[6]} for r in result] if result else []

def get_team_form(team_name, last_n_matches=5):
    """Get recent form for a team (W/D/L for last N matches)"""
    query = f"""
        SELECT season, match_date, home_team, away_team, full_time_result
        FROM epl_matches 
        WHERE home_team = '{team_name}' OR away_team = '{team_name}'
        ORDER BY match_date DESC
        LIMIT {last_n_matches}
    """
    
    matches = execute_query(query, 'soccer_stats')
    if not matches:
        return None
    
    form = []
    for match in matches:
        season, date, home, away, result = match
        if home == team_name:
            team_result = 'W' if result == 'H' else ('L' if result == 'A' else 'D')
        else:
            team_result = 'W' if result == 'A' else ('L' if result == 'H' else 'D')
        form.append(team_result)
    
    return {
        'team': team_name,
        'form': ''.join(form),
        'recent_matches': matches
    }

def get_season_leaders(season, category='points'):
    """Get season leaders by category: points, goals_scored, goals_conceded, wins"""
    
    if category == 'points':
        order_by = "points DESC, goal_difference DESC"
    elif category == 'goals_scored':
        order_by = "goals_for DESC"
    elif category == 'goals_conceded':
        order_by = "goals_against ASC"
    elif category == 'wins':
        order_by = "wins DESC"
    else:
        order_by = "points DESC"
    
    query = f"""
        SELECT team, 
               SUM(matches) as played,
               SUM(wins) as wins,
               SUM(draws) as draws, 
               SUM(losses) as losses,
               SUM(goals_for) as goals_for,
               SUM(goals_against) as goals_against,
               SUM(goals_for) - SUM(goals_against) as goal_difference,
               SUM(wins * 3 + draws) as points
        FROM (
            SELECT home_team as team,
                   COUNT(*) as matches,
                   SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
                   SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as losses,
                   SUM(full_time_home_goals) as goals_for,
                   SUM(full_time_away_goals) as goals_against
            FROM epl_matches 
            WHERE season = '{season}'
            GROUP BY home_team
            
            UNION ALL
            
            SELECT away_team as team,
                   COUNT(*) as matches,
                   SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
                   SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as losses,
                   SUM(full_time_away_goals) as goals_for,
                   SUM(full_time_home_goals) as goals_against
            FROM epl_matches 
            WHERE season = '{season}'
            GROUP BY away_team
        ) season_stats
        GROUP BY team
        ORDER BY {order_by}
    """
    
    result = execute_query(query, 'soccer_stats')
    return [{'position': i+1, 'team': r[0], 'played': r[1], 'wins': r[2], 'draws': r[3], 
             'losses': r[4], 'goals_for': r[5], 'goals_against': r[6], 
             'goal_difference': r[7], 'points': r[8]} for i, r in enumerate(result)] if result else []

def find_matches(home_team=None, away_team=None, season=None, date_from=None, date_to=None, 
                min_goals=None, max_goals=None, result=None, limit=50):
    """Flexible match finder with multiple filter options"""
    conditions = []
    params = []
    
    if home_team:
        conditions.append("home_team = ?")
        params.append(home_team)
    
    if away_team:
        conditions.append("away_team = ?")
        params.append(away_team)
    
    if season:
        conditions.append("season = ?")
        params.append(season)
    
    if date_from:
        conditions.append("match_date >= ?")
        params.append(date_from)
    
    if date_to:
        conditions.append("match_date <= ?")
        params.append(date_to)
    
    if min_goals:
        conditions.append("(full_time_home_goals + full_time_away_goals) >= ?")
        params.append(min_goals)
    
    if max_goals:
        conditions.append("(full_time_home_goals + full_time_away_goals) <= ?")
        params.append(max_goals)
    
    if result:
        conditions.append("full_time_result = ?")
        params.append(result)
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)
    
    query = f"""
        SELECT season, match_date, home_team, away_team, 
               full_time_home_goals, full_time_away_goals, full_time_result,
               home_shots, away_shots, home_yellow_cards, away_yellow_cards
        FROM epl_matches 
        {where_clause}
        ORDER BY match_date DESC 
        LIMIT ?
    """
    
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return [{'season': r[0], 'date': str(r[1]), 'home_team': r[2], 'away_team': r[3],
                'home_goals': r[4], 'away_goals': r[5], 'result': r[6],
                'home_shots': r[7], 'away_shots': r[8], 
                'home_yellows': r[9], 'away_yellows': r[10]} for r in results]
    except Exception as e:
        print(f"Error finding matches: {e}")
        return []

def get_team_matches(team_name, limit=10):
    """Get recent matches for a specific team"""
    query = """
        SELECT season, match_date, home_team, away_team, 
               full_time_home_goals, full_time_away_goals, full_time_result
        FROM epl_matches 
        WHERE home_team = ? OR away_team = ?
        ORDER BY match_date DESC 
        LIMIT ?
    """
    
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute(query, (team_name, team_name, limit))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        print(f"\nLast {len(results)} matches for {team_name}:")
        print("Date       | Season  | Matchup                      | Score | Result")
        print("-----------|---------|------------------------------|-------|-------")
        
        for match in results:
            season, date, home, away, home_goals, away_goals, result = match
            matchup = f"{home} vs {away}"
            score = f"{home_goals}-{away_goals}"
            # Determine result from team's perspective
            if home == team_name:
                team_result = "W" if result == "H" else ("L" if result == "A" else "D")
            else:
                team_result = "W" if result == "A" else ("L" if result == "H" else "D")
            
            print(f"{str(date):<10} | {season:<7} | {matchup:<28} | {score:<5} | {team_result}")
        
        return results
    except Exception as e:
        print(f"Error getting team matches: {e}")
        return []

def get_season_table(season):
    """Calculate league table for a specific season"""
    query = """
        SELECT team, 
               SUM(matches) as played,
               SUM(wins) as wins,
               SUM(draws) as draws, 
               SUM(losses) as losses,
               SUM(goals_for) as goals_for,
               SUM(goals_against) as goals_against,
               SUM(goals_for) - SUM(goals_against) as goal_difference,
               SUM(wins * 3 + draws) as points
        FROM (
            SELECT home_team as team,
                   COUNT(*) as matches,
                   SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
                   SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as losses,
                   SUM(full_time_home_goals) as goals_for,
                   SUM(full_time_away_goals) as goals_against
            FROM epl_matches 
            WHERE season = ?
            GROUP BY home_team
            
            UNION ALL
            
            SELECT away_team as team,
                   COUNT(*) as matches,
                   SUM(CASE WHEN full_time_result = 'A' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN full_time_result = 'D' THEN 1 ELSE 0 END) as draws,
                   SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as losses,
                   SUM(full_time_away_goals) as goals_for,
                   SUM(full_time_home_goals) as goals_against
            FROM epl_matches 
            WHERE season = ?
            GROUP BY away_team
        ) season_stats
        GROUP BY team
        ORDER BY points DESC, goal_difference DESC, goals_for DESC
    """
    
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute(query, (season, season))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        print(f"\nPremier League Table - {season}")
        print("Pos | Team           | P  | W  | D  | L  | GF | GA | GD  | Pts")
        print("----|----------------|----|----|----|----|----|----|-----|----")
        
        for i, row in enumerate(results, 1):
            team, played, wins, draws, losses, gf, ga, gd, points = row
            gd_str = f"+{gd}" if gd > 0 else str(gd)
            print(f"{i:2}  | {team:<14} | {played:2} | {wins:2} | {draws:2} | {losses:2} | {gf:2} | {ga:2} | {gd_str:>3} | {points:3}")
        
        return results
    except Exception as e:
        print(f"Error getting season table: {e}")
        return []

def search_matches(home_team=None, away_team=None, season=None, min_goals=None, result=None, limit=20):
    """Search for matches with various filters"""
    conditions = []
    params = []
    
    if home_team:
        conditions.append("home_team = ?")
        params.append(home_team)
    
    if away_team:
        conditions.append("away_team = ?")
        params.append(away_team)
    
    if season:
        conditions.append("season = ?")
        params.append(season)
    
    if min_goals:
        conditions.append("(full_time_home_goals + full_time_away_goals) >= ?")
        params.append(min_goals)
    
    if result:
        conditions.append("full_time_result = ?")
        params.append(result)
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)
    
    query = f"""
        SELECT season, match_date, home_team, away_team, 
               full_time_home_goals, full_time_away_goals, full_time_result
        FROM epl_matches 
        {where_clause}
        ORDER BY match_date DESC 
        LIMIT ?
    """
    
    try:
        conn = get_connection('soccer_stats')
        cur = conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        print(f"\nFound {len(results)} matches:")
        print("Date       | Season  | Home Team     | Away Team     | Score | Result")
        print("-----------|---------|---------------|---------------|-------|-------")
        
        for match in results:
            season, date, home, away, home_goals, away_goals, result = match
            score = f"{home_goals}-{away_goals}"
            result_desc = {"H": "Home", "A": "Away", "D": "Draw"}[result]
            print(f"{str(date):<10} | {season:<7} | {home:<13} | {away:<13} | {score:<5} | {result_desc}")
        
        return results
    except Exception as e:
        print(f"Error searching matches: {e}")
        return []

# ===== ANALYSIS & COMPARISON FUNCTIONS =====

def compare_teams(team1, team2, season=None):
    """Compare two teams head-to-head with detailed stats"""
    print(f"\n=== {team1} vs {team2} COMPARISON ===")
    
    # Get individual team stats
    team1_stats = get_team_stats(team1, season)
    team2_stats = get_team_stats(team2, season)
    
    if not team1_stats or not team2_stats:
        print("Error getting team stats")
        return
    
    print(f"\nSeason: {season or 'All time'}")
    print(f"{'Stat':<20} | {team1:<15} | {team2:<15}")
    print("-" * 55)
    print(f"{'Matches Played':<20} | {team1_stats['matches_played']:<15} | {team2_stats['matches_played']:<15}")
    print(f"{'Wins':<20} | {team1_stats['wins']:<15} | {team2_stats['wins']:<15}")
    print(f"{'Win %':<20} | {team1_stats['win_percentage']:<15}% | {team2_stats['win_percentage']:<15}%")
    print(f"{'Points':<20} | {team1_stats['points']:<15} | {team2_stats['points']:<15}")
    print(f"{'Goals Scored':<20} | {team1_stats['goals_scored']:<15} | {team2_stats['goals_scored']:<15}")
    print(f"{'Goals Conceded':<20} | {team1_stats['goals_conceded']:<15} | {team2_stats['goals_conceded']:<15}")
    print(f"{'Goal Difference':<20} | {team1_stats['goal_difference']:<15} | {team2_stats['goal_difference']:<15}")
    
    # Head-to-head record
    h2h = get_head_to_head(team1, team2, 10)
    if h2h:
        print(f"\nDirect meetings (last 10): {h2h['total_matches']} matches")
        print(f"{team1} wins: {h2h['team1_wins']}")
        print(f"{team2} wins: {h2h['team2_wins']}")
        print(f"Draws: {h2h['draws']}")

def analyze_team(team_name, season=None):
    """Complete analysis of a team's performance"""
    print(f"\n=== {team_name} ANALYSIS ===")
    
    # Basic stats
    stats = get_team_stats(team_name, season)
    if not stats:
        print("No data found for this team")
        return
    
    print(f"\nSeason: {stats['season']}")
    print(f"Matches: {stats['matches_played']}")
    print(f"Record: {stats['wins']}W-{stats['draws']}D-{stats['losses']}L")
    print(f"Win Rate: {stats['win_percentage']}%")
    print(f"Points: {stats['points']}")
    print(f"Goals: {stats['goals_scored']} scored, {stats['goals_conceded']} conceded")
    gd = stats['goal_difference']
    gd_str = f"+{gd}" if gd > 0 else str(gd)
    print(f"Goal Difference: {gd_str}")
    print(f"Average per match: {stats['avg_goals_scored']:.1f} scored, {stats['avg_goals_conceded']:.1f} conceded")
    
    # Recent form
    form = get_team_form(team_name, 5)
    if form:
        print(f"Recent form (last 5): {form['form']}")
    
    # Recent matches
    recent = get_team_matches(team_name, 5)
    if recent:
        print(f"\nLast 5 matches:")
        for match in recent:
            season, date, home, away, home_goals, away_goals, result = match
            score = f"{home_goals}-{away_goals}"
            if home == team_name:
                opponent = away
                team_result = "W" if result == "H" else ("L" if result == "A" else "D")
                venue = "vs"
            else:
                opponent = home
                team_result = "W" if result == "A" else ("L" if result == "H" else "D")
                venue = "@"
            print(f"  {team_result} {venue} {opponent} {score} ({date})")

def get_interesting_facts():
    """Get interesting facts and records from the database"""
    facts = []
    
    # Highest scoring match
    highest = get_high_scoring_matches(1, limit=1)
    if highest:
        match = highest[0]
        facts.append(f"Highest scoring match: {match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']} ({match['total_goals']} goals, {match['season']})")
    
    # Most goals in a season
    top_scorers = get_top_scorers(limit=1)
    if top_scorers:
        team = top_scorers[0]
        facts.append(f"Most prolific team overall: {team['team']} with {team['total_goals']} goals in {team['matches']} matches")
    
    # Best defense
    best_defense = get_best_defenses(limit=1)
    if best_defense:
        team = best_defense[0]
        facts.append(f"Best defense overall: {team['team']} conceded only {team['total_conceded']} goals in {team['matches']} matches")
    
    return facts

def explore():
    """Interactive function to explore the database easily"""
    print("\n🔍 SOCCER STATS EXPLORER")
    print("=" * 30)
    
    # Show some quick options
    print("\nWhat would you like to explore?")
    print("1. Team analysis")
    print("2. Season comparison") 
    print("3. High-scoring matches")
    print("4. Head-to-head records")
    print("5. Recent form")
    
    print(f"\nAvailable teams: {', '.join(get_all_teams()[:10])}... (+{len(get_all_teams())-10} more)")
    print(f"Available seasons: {', '.join(get_all_seasons()[-5:])}")
    
    print("\n💡 Try these commands:")
    print("• analyze_team('Arsenal')")
    print("• compare_teams('Arsenal', 'Tottenham')")
    print("• get_season_table('2023/24')")
    print("• get_high_scoring_matches(5)")
    print("• find_matches(season='2023/24', min_goals=4)")

def show_available_functions():
    """Show all available no-SQL functions"""
    print("\n🚀 NO-SQL DATA ACCESS FUNCTIONS")
    print("=" * 50)
    print("\n📊 BASIC DATA:")
    print("• get_all_teams() - List all teams")
    print("• get_all_seasons() - List all seasons") 
    print("• get_team_stats('Arsenal') - Complete team stats")
    print("• get_team_stats('Arsenal', '2023/24') - Team stats for specific season")
    
    print("\n🏆 LEAGUE TABLES & RANKINGS:")
    print("• get_season_table('2023/24') - Full league table")
    print("• get_season_leaders('2023/24', 'points') - Season leaders by category")
    print("• get_top_scorers() - Highest scoring teams")
    print("• get_best_defenses() - Best defensive teams")
    
    print("\n⚽ MATCH DATA:")
    print("• get_team_matches('Liverpool', 10) - Last 10 matches for team")
    print("• get_high_scoring_matches(6) - Matches with 6+ goals")
    print("• find_matches(home_team='Arsenal', season='2023/24') - Flexible search")
    print("• get_head_to_head('Arsenal', 'Tottenham') - Direct comparison")
    
    print("\n📈 ANALYSIS:")
    print("• analyze_team('Manchester City') - Complete team analysis")
    print("• compare_teams('Arsenal', 'Chelsea') - Head-to-head comparison")
    print("• get_team_form('Liverpool', 5) - Recent form")
    print("• get_interesting_facts() - Fun facts and records")
    print("• explore() - Interactive exploration guide")
    
    print("\n💡 EXAMPLES:")
    print("find_matches(season='2023/24', min_goals=4, limit=10)")
    print("get_season_leaders('2022/23', 'goals_scored')")
    print("compare_teams('Man City', 'Liverpool', '2023/24')")
    print("\n" + "=" * 50)

def run_all_examples():
    """Run all example queries to demonstrate database usage"""
    print("\nSOCCER STATS DATABASE - LEARNING EXAMPLES")
    print("=" * 60)
    
    example_basic_queries()
    example_aggregation_queries()
    example_advanced_queries()
    
    # Show new no-SQL functions
    show_available_functions()

if __name__ == "__main__":
    print("MariaDB Soccer Stats Database Setup")
    print("=" * 40)
    
    # Test connection
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        version = cur.fetchone()
        print(f"MariaDB version: {version[0]}")
        cur.close()
        conn.close()
        
        # Setup soccer database
        print("\nSetting up soccer_stats database...")
        if setup_soccer_database():
            
            # Check if data exists
            result = execute_query("SELECT COUNT(*) FROM epl_matches", 'soccer_stats')
            if result and result[0][0] > 0:
                print(f"Database contains {result[0][0]} matches - ready to use!")
            else:
                print("Database is empty. Use restore_from_backup() if needed.")
            
            # Show available functions instead of running all examples
            show_available_functions()
            
            # Quick demo
            print("\n🎯 QUICK DEMO:")
            teams = get_all_teams()
            print(f"Database contains {len(teams)} teams across {len(get_all_seasons())} seasons")
            
            # Show Arsenal stats as example
            stats = get_team_stats('Arsenal', '2023/24')
            if stats:
                print(f"Arsenal 2023/24: {stats['wins']}W-{stats['draws']}D-{stats['losses']}L, {stats['points']} pts")
        else:
            print("Failed to setup database")
        
    except mariadb.Error as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
