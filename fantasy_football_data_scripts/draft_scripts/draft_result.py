# draft_result.py

class DraftResult:
    def __init__(self, extracted_data):
        self.cost = extracted_data.get("cost", "N/A")
        self.pick = extracted_data.get("pick", "N/A")
        self.round = extracted_data.get("round", "N/A")
        self.team_key = extracted_data.get("team_key", "N/A")
        self.player_id = extracted_data.get("player_id", "")
        self.year = None
        self.name_full = extracted_data.get("name_full", "N/A")
        self.primary_position = extracted_data.get("primary_position", "N/A")
        self.average_pick = extracted_data.get("average_pick", "N/A")
        self.average_round = extracted_data.get("average_round", "N/A")
        self.average_cost = extracted_data.get("average_cost", "N/A")
        self.percent_drafted = extracted_data.get("percent_drafted", "N/A")
        self.is_keeper_status = extracted_data.get("is_keeper_status", "N/A")
        self.is_keeper_cost = extracted_data.get("is_keeper_cost", "N/A")