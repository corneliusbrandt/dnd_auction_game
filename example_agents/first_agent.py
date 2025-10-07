import numpy as np
import random
import os
from dnd_auction_game import AuctionGameClient
import matplotlib.pyplot as plt

############################################################################################
#
# first_agent self developed
#
############################################################################################

class FirstAgent:
    def __init__(self):
        pass

    def expected_value(self, auction:dict) -> float:
        """
        Calculates the expected value for a roll in the format NdM + Bonus (e.g., 3d6 + 2).
        """
        e = (auction["num"] * ((auction["die"] + 1)/2)) + auction["bonus"]
        return e
    
    def variance(self, auction:dict) -> float:
        """
        Calculates the variance for a roll in the format NdM + Bonus (e.g., 3d6 + 2).
        """
        die = auction["die"]
        num = auction["num"]
        variance = num * ((die**2 - 1) / 12)
        return variance

    def pmf_ndm(self, num, die, bonus=0):
        """
        Calculates the probability mass function (PMF)
        for a roll in the format NdM + Bonus (e.g., 3d6 + 2).
        
        Parameters:
            num   - number of dice (N)
            die   - number of sides on the die (M)
            bonus - fixed bonus value, can also be negative
        
        Returns:
            dict: {point value: probability}
        """
        # Base distribution of a single die: uniformly distributed from 1 to die
        single = np.ones(die) / die  # e.g. [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

        # Convolution for N dice = sum of distributions
        dist = single
        for _ in range(num - 1):
            dist = np.convolve(dist, single)

        # possible sums of the N dice range from N to N*die
        values = np.arange(num, num * die + 1) + bonus

        # PMF as a dictionary
        return dict(zip(values, dist))

    def evaluate_downside_risk(self, auction:dict, threshold, risk_factor) -> float:
        """
        Evaluates the downside risk of an auction using a threshold and risk factor.
        """
        # Calculate the probability mass function for the auction
        pmf = self.pmf_ndm(auction["num"], auction["die"], auction["bonus"])
        # Calculate the expected risk of being below the threshold
        downside_risk = sum(max(0, threshold - x) * p for x, p in pmf.items())
        # evaluate the utility of the auction based on a risk factor and downside risk
        # the higher the risk factor, the more konservative the agent will be
        utility = auction["expected_value"] - risk_factor * downside_risk

        return utility, downside_risk

    def get_auctions(self, auctions:dict):
        auctions_list = []
        for auction_id, auction in auctions.items():
            auctions_list.append(auction)
            auction["expected_value"] = self.expected_value(auction)
            auction["std_dev"] = self.variance(auction) ** 0.5
            auction["id"] = auction_id
        return auctions_list
    
    def get_wanted_auctions(self, auctions:list, min_utility, max_utility):
        wanted_auctions = []
        sorted_auctions = sorted(auctions, key=lambda x: x["utility"], reverse=True)
        for auction in sorted_auctions:
            if min_utility <= auction["utility"] <= max_utility:
                wanted_auctions.append(auction)
        return wanted_auctions

#############################################################################################

    def bid(self, agent_id:str, current_round:int, states:dict, auctions:dict, prev_auctions:dict, bank_state:dict):
        agent_state = states[agent_id]
        current_gold = agent_state["gold"]
        points = agent_state["points"]

        # get auction parameters
        auctions_list = self.get_auctions(auctions)
        for auction in auctions_list:
            # Threshold for dowsnside risk and risk factor for utility calculation
            utility, downside_risk = self.evaluate_downside_risk(auction, threshold=5, risk_factor=0.5)
            auction["downside_risk"] = downside_risk
            auction["utility"] = utility
            

        next_round_gold_income = 0
        if len(bank_state["gold_income_per_round"]) > 0:
            next_round_gold_income = bank_state["gold_income_per_round"][0]

        auctions_to_bid = self.get_wanted_auctions(auctions_list, min_utility=15, max_utility=50)
        bids = {}
        for auction in auctions_to_bid:
            bid_amount = 300
            if current_gold > 2000:
                bids[auction["id"]] = bid_amount
                current_gold -= bid_amount

        return bids

############################################################################################

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

