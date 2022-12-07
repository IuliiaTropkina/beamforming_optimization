a = [1,2,3,4,5,6,7,8,9]

for i in range(1, int(len(a)/2)+1):
    a.pop(i)
print(a)