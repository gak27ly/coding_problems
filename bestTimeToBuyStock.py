'''
股票交易类问题：
确认是否可以不进行交易，交易次数限制， 同一天买卖是否可行
'''

'''
149. Best Time to Buy and Sell Stock
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction 
(ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
解法： 按照顺序，不断寻找最小股价，再求出当前最大利润。 注意初始值的设置。
'''
def maxProfit(self, prices):
    # write your code here
    minPrice = sys.maxsize
    maxProfit = 0
    
    for price in prices:
        
        maxProfit = max(price - minPrice, maxProfit)
        minPrice = min(minPrice, price)
    
    return maxProfit

'''
150. Best Time to Buy and Sell Stock II
Given an array prices, which represents the price of a stock in each day.
You may complete as many transactions as you like 
(ie, buy one and sell one share of the stock multiple times). 
However, you may not engage in multiple transactions at the same time 
(ie, if you already have the stock, you must sell it before you buy again).
Design an algorithm to find the maximum profit.
解法： 只要有价格上涨就计算出差价并加入总的profit中，保证所有的上涨都能赚到钱
'''

def maxProfit(self, prices):
    # write your code here
    
    minPrice = sys.maxsize
    maxProfit = 0
    
    for price in prices:
        if price - minPrice > 0:
            maxProfit += price - minPrice
            minPrice = sys.maxsize
        if price < minPrice:
            minPrice = price

    return maxProfit
'''
151. Best Time to Buy and Sell Stock III
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete at most two transactions.
解法： DP 两种选择中选择最大的：1. 不做交易，利润最大值与前一天相同交易量的利润相同 
							2. 遍历当前交易日之前的购入日m，交易利润最大值为 
							   当天股价price[j] - 购入日股价prices[m] + 购入日(交易量-1)的最大利润 dp[i-1][m]
'''
def maxProfit(self, prices):
    # write your code here
    if not prices:
        return 0
    n = len(prices)
    k = 2
    
    dp = [[0 for _ in range(n)] for _ in range(k + 1)]

    for i in range(1, k + 1):
    	tradeProfit = -prices[0]
        for j in range(1, n):
            noTradeProfit = dp[i][j - 1]
            
            for m in range(j):
                tradeProfit = max(prices[j] - prices[m] + dp[i - 1][m], tradeProfit)
            
            dp[i][j] = max(noTradeProfit, tradeProfit)
            
    return dp[k][n - 1]

'''
使用maxdiff 来简化计算交易日的最大利润
始终记录之前交易日产生的最大的差值
maxDiff = max(maxDiff, dp[i-1][j-1] - price[j-1])
'''

def maxProfit(self, prices):
    # write your code here
    if not prices:
        return 0
    n = len(prices)
    k = 2
    dp = [[0] * n for _ in range(k + 1)]

    for i in range(1, k + 1):
		#必须在每一行之前重新设置maxDiff
        maxDiff = -sys.maxsize - 1

        for j in range(1, n):
            
            maxDiff = max(maxDiff, dp[i - 1][j - 1] - prices[j - 1])
            noTradeProfit = dp[i][j - 1]
            
            tradeProfit = prices[j] + maxDiff
            
            dp[i][j] = max(noTradeProfit, tradeProfit)
            
    return dp[k][n - 1]


def maxProfit(self, K, prices):
    # write your code here
    if not prices:
        return 0 
    
    n = len(prices)
    profit = 0
    #对于交易次数多于prices长度一半情况，只要有升值就可交易
    if K >= n // 2:
        for i in range(1, n):
            if prices[i] < prices[i - 1]:
                continue
            profit += prices[i] - prices[i -1]
        return profit
        
    dp = [[0] * n for _ in range(2)]
    #应为只要使用到上一行的数据，所以只要用 i%2 来进行space上的优化
    for i in range(1, K + 1):
        maxDiff = -prices[0]
        for j in range(1, n):
            maxDiff = max(dp[i % 2 - 1][j - 1] - prices[j - 1], maxDiff)
            dp[i % 2][j] = max(dp[i % 2][j - 1], prices[j] + maxDiff)
    return dp[K % 2][n - 1]






