import reader
import params
import matplotlib.pyplot as plt


x = []
total = 0
min_price = 1000
max_price = 0
for k in reader.parse(params.metadata_path):
    if 'price' in k:
        x.append(k['price'])
        total += 1
        if k['price'] > max_price:
            max_price = k['price']
        if k['price'] < min_price:
            min_price = k['price']
        if total % 10000 == 0:
            print 'Read ', total

print 'Total items : ', total
print 'Min price : ', min_price
print 'Max price : ', max_price

print 'Plotting histogram'
num_bins = 1000
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=0, facecolor='green')

plt.xlabel('Price range (USD)')
plt.ylabel('Number of products')
plt.title(r'Histogram of number of products vs price range')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.savefig('hist.jpg')
print 'Plot saved to hist.jpg'


