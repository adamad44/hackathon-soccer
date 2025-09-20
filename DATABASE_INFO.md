# Soccer Stats Database - Setup Complete! ⚽

## Database Location

Your MariaDB database is permanently stored at:

```
C:\Program Files\MariaDB 11.8\data\soccer_stats\
```

### Files:

- **epl_matches.ibd** (10MB) - Your actual data (9,380 matches)
- **epl_matches.frm** - Table structure
- **db.opt** - Database configuration

## Local Backup

- **soccer_stats_backup.sql** (1.8MB) - Complete database backup in your project folder

## 🗑️ Safe to Delete

You can now **safely delete** `epl_final.csv` - all data is permanently stored in MariaDB!

## Database Contents

- **9,380 EPL matches** from 2000-2025
- **25 seasons** of historical data
- **22 data points** per match (goals, shots, cards, fouls, corners, etc.)

## Quick Start Functions

```python
# Get team's recent matches
get_team_matches('Arsenal', 10)

# Get league table for a season
get_season_table('2023/24')

# Search for specific matches
search_matches(home_team='Liverpool', season='2022/23')
search_matches(min_goals=6)  # High-scoring games
```

## 🔐 Security Setup

The database connection is now secured using environment variables:

### Configuration Files:

- **`.env`** - Contains your actual credentials (NEVER commit to git)
- **`.env.example`** - Template for other developers
- **`.gitignore`** - Protects .env from being committed

### Environment Variables:

```
DB_USER=root
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=3306
```

## Connection Info

- **Host:** localhost (from .env)
- **Port:** 3306 (from .env)
- **Database:** soccer_stats
- **Table:** epl_matches
- **Credentials:** Stored securely in .env file

## Service Management

```powershell
# Check MariaDB service status
Get-Service -Name "MariaDB"

# Stop service (requires admin)
Stop-Service -Name "MariaDB"

# Start service (requires admin)
Start-Service -Name "MariaDB"
```

Your database is now ready for analysis, learning SQL, and building applications! 🚀
