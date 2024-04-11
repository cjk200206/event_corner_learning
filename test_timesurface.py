import numpy as np

segment1 = np.arange(-40, -20)  # (-20, -10) 区间内的整数
segment2 = np.arange(20, 40)  
choices = np.concatenate((segment1, segment2))
random_number = [np.random.choice(choices),np.random.choice(choices)]

print(random_number)