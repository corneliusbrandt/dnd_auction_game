import numpy as np
import random
import os
from dnd_auction_game import AuctionGameClient

############################################################################################
#
# first_agent self developed
#
############################################################################################

class FirstAgent:
    def __init__(self):
        pass

    def expected_value(self, auction:dict) -> float:
        e = (auction["num"] * ((auction["die"] + 1)/2)) + auction["bonus"]
        return e
    
    def variance(self, auction:dict) -> float:
        die = auction["die"]
        num = auction["num"]
        variance = num * ((die**2 - 1) / 12)
        return variance
   

    def evaluate_downside_risk(self, auction:dict) -> float:
        pass

    def get_auctions(self, auctions:dict):
        auctions_list = []
        for auction_id, auction in auctions.items():
            auctions_list.append(auction)
            auction["expected_value"] = self.expected_value(auction)
            auction["std_dev"] = self.variance(auction) ** 0.5
            auction["id"] = auction_id
        return auctions_list

    def bid(self, agent_id:str, current_round:int, states:dict, auctions:dict, prev_auctions:dict, bank_state:dict):
        agent_state = states[agent_id]
        current_gold = agent_state["gold"]
        points = agent_state["points"]
        auctions_list = self.get_auctions(auctions)

        next_round_gold_income = 0
        if len(bank_state["gold_income_per_round"]) > 0:
            next_round_gold_income = bank_state["gold_income_per_round"][0]   

        print(auctions_list)
        return {}


if __name__ == "__main__":
    
    host = "localhost"
    agent_name = "{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    player_id = "id_of_human_player"
    port = 8000

    game = AuctionGameClient(host=host,
                                agent_name=agent_name,
                                player_id=player_id,
                                port=port)
    agent = FirstAgent()
    try:
        game.run(agent.bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")

