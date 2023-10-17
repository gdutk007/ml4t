""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np		  	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return "gdutka3"  # replace tb34 with your Georgia Tech username.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def gtid():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return 903890585  # replace with your GT ID number  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  		 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    result = False  		  	   		  		 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		  		 		  		  		    	 		 		   		 		  
        result = True  		  	   		  		 		  		  		    	 		 		   		 		  
    return result  		  	   		  		 		  		  		    	 		 		   		 		  

MININT = -2**31  	   		  		 		  		  		    	 		 		   		 		  
MAXINT = 2**31-1

def run_episode(bet_amount):
    iterations = 0
    win_prob = 0.474 # TODO: THIS SHOULD BE BASED ON ROULETTE WHEEL 
    episode_winnings = 0
    winnings = np.zeros([1001], dtype=int)
    bet_amount = bet_amount
    while episode_winnings < 80 or iterations < 1000:
        won = False
        # if we reached 80, we forward fill
        if episode_winnings >= 80:
            bet_amount = 0
        while not won and iterations < 1000:
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
                bet_amount = 1
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2   
            winnings[iterations+1] = episode_winnings
            iterations += 1

    return winnings

def run_episode_with_bankroll(bet_amount):
    iterations = 0
    win_prob = 0.474 # TODO: THIS SHOULD BE BASED ON ROULETTE WHEEL 
    episode_winnings = 0
    winnings = np.zeros([1001], dtype=int)
    bet_amount = bet_amount
    while episode_winnings < 80 or iterations < 1000:
        won = False
        # if we reached 80, we forward fill
        if episode_winnings >= 80:
            bet_amount = 0
        while not won and iterations < 1000:
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
                bet_amount = 1
            else:
                episode_winnings = episode_winnings - bet_amount
                money_remaining = 256 + episode_winnings
                if bet_amount > money_remaining:
                    bet_amount = money_remaining
                else:
                    bet_amount = bet_amount * 2 
            winnings[iterations+1] = episode_winnings
            iterations += 1
            if episode_winnings <= -256:
                # iterations = 999
                #print("out of money!")
                return winnings
                # break

    return winnings

def write_images_to_file(fig1, fig2, fig3, fig4, fig5):
    tmpfile = BytesIO()
    fig1.savefig(tmpfile, format='png')
    fig1.savefig('./images/fig1.png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

    tmpfile = BytesIO()
    fig2.savefig(tmpfile, format='png')
    fig2.savefig('./images/fig2.png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format='png')
    fig3.savefig('./images/fig3.png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

    tmpfile = BytesIO()
    fig4.savefig(tmpfile, format='png')
    fig4.savefig('./images/fig4.png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

    tmpfile = BytesIO()
    fig5.savefig(tmpfile, format='png')
    fig5.savefig('./images/fig5.png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br />'

    with open('./images/p1_result.html','w') as f:
        f.write(html)

def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    # win_prob = 0.60  # set appropriately to the probability of a win  		  	   		  		 		  		  		    	 		 		   		 		  
    np.random.seed(gtid())  # do this only once  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments  		  	   		  		 		  		  		    	 		 		   		 		  
    figure_one_results = np.zeros([10,1001])
    for i in range(10):
        bet_amount = 1
        figure_one_results[i] = run_episode(bet_amount)
    
    x = np.arange(0,1001)
    fig1, ax1 = plt.subplots()    
    ax1.set_ylim([-256,100])
    ax1.set_xlim([0,300])
    ax1.set_title("Roulete gains per 1000 games") 
    ax1.set_xlabel("spin number") 
    ax1.set_ylabel("cumulative winnings")
    label_names = []
    
    for i in range(10): 
        ax1.plot(x,figure_one_results[i])
        label_names += ["Episode " + str(i)]
    ax1.legend(label_names)
    fig2, ax2 = plt.subplots()
    ax2.set_xlim([0,300])
    ax2.set_ylim([-256,100])
    ax2.set_title("Mean Roulete gains per 1000 games") 
    ax2.set_xlabel("spin number") 
    ax2.set_ylabel("cumulative winnings")
    # label_names = []
    figure_two_results = np.zeros([1000,1001])
    for i in range(1000):
        bet_amount = 1
        figure_two_results[i] = run_episode(bet_amount)
    figure_two_mean = figure_two_results.mean(axis=0)
    figure_two_std = figure_two_results.std(axis=0)
    figure_two_median = np.median(figure_two_results, axis=0)
    upper = figure_two_mean + figure_two_std*2
    lower = figure_two_mean - figure_two_std*2 
    ax2.plot(x,figure_two_mean)
    ax2.plot(x,upper)
    ax2.plot(x,lower)
    ax2.legend(['mean', 'upper band', 'lower band'])
    # import pdb; pdb.set_trace()
    fig3, ax3 = plt.subplots()
    ax3.set_xlim([0,300])
    ax3.set_ylim([-256,100])
    ax3.set_title("Median Roulete gains per 1000 games") 
    ax3.set_xlabel("spin number") 
    ax3.set_ylabel("cumulative winnings")
    ax3.plot(x,figure_two_median)
    ax3.plot(x,upper)
    ax3.plot(x,lower)
    ax3.legend(['median', 'upper band', 'lower band'])
    


    ###### part 2 #######

    fig4, ax4 = plt.subplots()
    ax4.set_xlim([0,300])
    ax4.set_ylim([-256,100])
    ax4.set_title("Mean Roulete gains per 1000 games with bankroll") 
    ax4.set_xlabel("spin number") 
    ax4.set_ylabel("cumulative winnings")
    figure_two_results = np.zeros([1000,1001])
    for i in range(1000):
        figure_two_results[i] = run_episode_with_bankroll(1)
    figure_two_mean = figure_two_results.mean(axis=0)
    figure_two_std = figure_two_results.std(axis=0)
    figure_two_median = np.median(figure_two_results, axis=0)
    upper = figure_two_mean + figure_two_std*2
    lower = figure_two_mean - figure_two_std*2 
    ax4.plot(x,figure_two_mean)
    ax4.plot(x,upper)
    ax4.plot(x,lower)
    ax4.legend(['mean', 'upper band', 'lower band'])

    fig5, ax5 = plt.subplots()
    ax5.set_xlim([0,300])
    ax5.set_ylim([-256,100])
    ax5.set_title("Median Roulete gains per 1000 games with bankroll") 
    ax5.set_xlabel("spin number") 
    ax5.set_ylabel("cumulative winnings")
    ax5.plot(x,figure_two_median)
    ax5.plot(x,upper)
    ax5.plot(x,lower)
    ax5.legend(['median', 'upper band', 'lower band'])

    # plt.show()
    # import pdb; pdb.set_trace()
    write_images_to_file(fig1,fig2,fig3,fig4,fig5)


if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
