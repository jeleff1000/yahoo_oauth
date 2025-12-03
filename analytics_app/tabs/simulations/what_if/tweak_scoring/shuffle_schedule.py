import numpy as np


def shuffle_schedule(df):
    """Optimized schedule shuffling with vectorized operations."""
    # Get unique managers and weeks efficiently
    managers = df["manager"].unique()
    weeks = sorted(df["week"].unique())

    num_managers = len(managers)
    schedule = []

    # Shuffle managers once
    np.random.shuffle(managers)

    # Generate round-robin schedule
    for _ in weeks:
        round_robin = []
        for i in range(num_managers // 2):
            round_robin.append((managers[i], managers[num_managers - 1 - i]))
        managers = np.roll(managers, 1)
        schedule.append(round_robin)

    # Shuffle weeks
    np.random.shuffle(weeks)

    # Build final schedule as a list
    final_schedule = []
    for week, matchups in zip(weeks, schedule):
        for match in matchups:
            final_schedule.append((week, match[0], match[1]))

    # Initialize win/loss columns
    df["Sim_Wins"] = 0
    df["Sim_Losses"] = 0

    # Create a lookup dictionary for faster point retrieval
    week_manager_points = df.set_index(["week", "manager"])["team_points"].to_dict()

    # Process all matchups efficiently
    updates = {"wins": [], "losses": []}

    for week, manager1, manager2 in final_schedule:
        team1_points = week_manager_points.get((week, manager1))
        team2_points = week_manager_points.get((week, manager2))

        if team1_points is not None and team2_points is not None:
            if team1_points > team2_points:
                updates["wins"].append((manager1, week))
                updates["losses"].append((manager2, week))
            elif team1_points < team2_points:
                updates["losses"].append((manager1, week))
                updates["wins"].append((manager2, week))

    # Vectorized update of wins
    for manager, week in updates["wins"]:
        mask = (df["manager"] == manager) & (df["week"] == week)
        df.loc[mask, "Sim_Wins"] += 1

    # Vectorized update of losses
    for manager, week in updates["losses"]:
        mask = (df["manager"] == manager) & (df["week"] == week)
        df.loc[mask, "Sim_Losses"] += 1

    return df
