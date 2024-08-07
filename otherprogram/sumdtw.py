import os
import numpy as np

path = '/home/hhc102u/SDT/Generated/2sets_test_debug/test/dtwlog.txt'

#ICDAR 2013: 60 writers (authors)
writer_total_num = 60
sdt=[0] * writer_total_num
g1=[0] * writer_total_num
i = [0] * writer_total_num
dtw_s=[0] * writer_total_num # g1<sdt


with open(path, 'r') as file:
    for line in file:
        temp = line.split(':')
        
        w_i = int(temp[0].split('_')[0])
        
        v1 = float(temp[2].split(',')[0])
        v2 = float(temp[3].split(',')[0])
        sdt[w_i]+=v1
        g1[w_i]+=v2
        i[w_i]+=1
        if v1 >= v2:
            dtw_s[w_i] += 1

sdt_np = np.array(sdt)
g1_np = np.array(g1)
i_np = np.array(i)
dtw_s_np = np.array(dtw_s)

dtw_s_rate = dtw_s_np/i_np
dtw_s_rate_num = sum(dtw_s_rate > 0.5)
dtw_s_rate_num2 = sum(dtw_s_rate == 0.5)
result1 = np.divide(sdt_np, i_np)
result2 = np.divide(g1_np, i_np)
result3 = result1 > result2
count = np.count_nonzero(result3)

result4 = sum(result1)/writer_total_num
result5 = sum(result2)/writer_total_num

result_minus = result1 - result2
#print(i)
#print(sdt)
#print(g1)

print(result4)
print(result5)
print(np.round(dtw_s_rate, decimals=2))
print('test class %s: avg_win:%s, samples_number_win:%s, samples_number_even:%s' %(writer_total_num, count, dtw_s_rate_num, dtw_s_rate_num2))