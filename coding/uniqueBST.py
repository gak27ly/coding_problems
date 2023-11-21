Unique Binary Search Trees

def numTrees(self, n):
    # write your code here
    if n is None:
        return 0
    if n == 0:
        return 1
        
    dp = [0] * (n+1)
    
    dp[0] = 1
    dp[1] = 1
        
        
    
    for i in range(2, n+1):
        for j in range(i):
            if i - j - 1 < 0:
                continue
            dp[i] += dp[j] * dp[i - j - 1]
            
    
    return dp[n]

