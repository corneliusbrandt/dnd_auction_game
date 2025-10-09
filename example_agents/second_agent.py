import numpy as np
import random
import os
from dnd_auction_game import AuctionGameClient
import matplotlib.pyplot as plt
import json

############################################################################################
#
# first_agent self developed
#
############################################################################################

class FirstAgent:
    def __init__(self):
        # Plotting attributes
        self.lines = None    
        self.rounds = []      
        self.money = []        
        self.points = [] 

        # Load parameters from a JSON file to change them during the game
        self.param_file = "agent_params.json"
        if not os.path.exists(self.param_file):
            with open(self.param_file, "w") as f:
                json.dump({"threshold": 5, "risk_factor": 0.5, "min_utility": 5, "max_utility": 20}, f)
        self.load_params()

    def load_params(self):
        with open(self.param_file, "r") as f:
            params = json.load(f)
        self.threshold = params["threshold"]
        self.min_utility = params["min_utility"]
        self.max_utility = params["max_utility"]
        self.risk_factor = params["risk_factor"]

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
        best_2_auctions = []
        sorted_auctions = sorted(auctions, key=lambda x: x["utility"], reverse=True)

        # Get auctions within the desired utility range
        for auction in sorted_auctions:
            if min_utility <= auction["utility"] <= max_utility:
                wanted_auctions.append(auction)

        # Limit to 5 auctions if more are wanted
        if len(wanted_auctions) > 5:
            wanted_auctions = random.sample(wanted_auctions, 5)

        best_2_auctions = sorted_auctions[:2]
        return wanted_auctions, best_2_auctions

    def live_plot_rounds(self, x, y1, y2, colors=('g', 'b'),
                        xlabel='Round', ylabel='Wert', title='Live-Tracking: Money & Points', lines=None):
        """
        Creates a live plot for tracking Money and Points over rounds.
        """
        if lines is None:
            plt.ion()
            # Zwei Subplots untereinander (2 Reihen, 1 Spalte)
            self.fig, (self.ax_money, self.ax_points) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            plt.subplots_adjust(hspace=0.3)

            # Money-Plot
            line_money, = self.ax_money.plot(x, y1, colors[0] + '-', label='Money')
            self.ax_money.set_ylabel('Gold')
            self.ax_money.set_title('Money over Rounds')
            self.ax_money.legend()

            # Points-Plot
            line_points, = self.ax_points.plot(x, y2, colors[1] + '-', label='Points')
            self.ax_points.set_xlabel(xlabel)
            self.ax_points.set_ylabel('Points')
            self.ax_points.set_title('Points over Rounds')
            self.ax_points.legend()

            # Textbox oben rechts im Points-Plot für Effizienz
            self.eff_text = self.ax_points.text(0.98, 0.95, "", transform=self.ax_points.transAxes,
                                                fontsize=10, verticalalignment='top', horizontalalignment='right',
                                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.show()
            self.lines = (line_money, line_points)

        else:
            line_money, line_points = self.lines
            line_money.set_xdata(x)
            line_money.set_ydata(y1)
            line_points.set_xdata(x)
            line_points.set_ydata(y2)

            self.ax_money.relim()
            self.ax_money.autoscale_view()
            self.ax_points.relim()
            self.ax_points.autoscale_view()

        # --- Effizienz (Points pro Gold über letzte 10 Runden) ---
        efficiency = 0
        sum_points = sum(self.points[-10:])
        sum_gold = sum(self.money[-10:])
        if sum_gold > 0:
            efficiency = sum_points / sum_gold

        self.eff_text.set_text(f"Points/Gold (10R): {efficiency:.3f}")

        plt.pause(0.05)

#############################################################################################

    def bid(self, agent_id:str, current_round:int, states:dict, auctions:dict, prev_auctions:dict, bank_state:dict):
        agent_state = states[agent_id]
        current_gold = agent_state["gold"]
        points = agent_state["points"]
        self.load_params()

        # get auction parameters
        auctions_list = self.get_auctions(auctions)
        for auction in auctions_list:
            # Threshold for dowsnside risk and risk factor for utility calculation
            utility, downside_risk = self.evaluate_downside_risk(auction, threshold=self.threshold, risk_factor=self.risk_factor)
            auction["downside_risk"] = downside_risk
            auction["utility"] = utility

        # Bid mechanism
        auctions_to_bid, best_2_auctions = self.get_wanted_auctions(auctions_list, min_utility=self.min_utility, max_utility=self.max_utility)
        bids = {}
        if current_round % 100 == 0:
            bid_amount = 0.30 * current_gold
            for auction in best_2_auctions:
                if current_gold > bid_amount:
                    bids[auction["id"]] = int(bid_amount)
                    current_gold -= int(bid_amount)
        elif current_round >= 994:
            bid_amount = 0.20 * current_gold
            for auction in best_2_auctions[:1]:
                    bids[auction["id"]] = int(bid_amount)
                    current_gold -= int(bid_amount)
        elif best_2_auctions[0]["utility"] > 40 and current_gold > 3000:
            bid_amount = 0.60 * current_gold
            for auction in best_2_auctions[:1]:
                if current_gold > bid_amount:
                    bids[auction["id"]] = int(bid_amount)
                    current_gold -= int(bid_amount)
        else:
            for auction in auctions_to_bid:
                bid_amount = ((0.50 * current_gold)/len(auctions_to_bid)) * (auction["utility"] / auction["expected_value"])
                if current_gold > 2000:
                    bids[auction["id"]] = bid_amount
                    current_gold -= bid_amount

        #Plotting
        self.rounds.append(current_round)
        self.money.append(current_gold)
        self.points.append(points)
        self.live_plot_rounds(self.rounds, self.money, self.points, lines=self.lines)

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

    # Keep plot open
    plt.ioff()
    plt.show()

