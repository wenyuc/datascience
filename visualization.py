from matplotlib import pyplot as plt

# single-line chart
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

assert len(years) == len(gdp), "length of two lists should be same length"

# create a line chart, years in x-axis, gdp on y-axis
plt.plot(years, gdp, color = 'red', marker = 'x', linestyle = 'solid')

# add a title
plt.title('Nominal GDP')

# add a label to y-axis
plt.ylabel("Billions of $")
#plt.show()
plt.savefig('im/years_gdp.png')
plt.close()

# bar chart
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# plot bars with left x-coordinates [0,1,2,3,4], heights[num_oscars]
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movie")       # add a title
plt.ylabel("# of Academy Awards")    # label the y-axis

# x-axis
plt.xticks(range(len(movies)), movies)
#plt.show()
plt.savefig('im/movies_oscars.png')
plt.close()

# histograms of bucketed numeric values
from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Bucket grades by decile, but put 100 in with the 90s
Histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)
#print(Histogram)

plt.bar([x + 5 for x in Histogram.keys()], # shifts bars right by 5
         Histogram.values(),  # given each bar its correct heights
         10,                   # given each bar a width of 10
         edgecolor = (0,0,0))  # black edges for each bar

plt.axis([-5, 105, 0, 5]) # x-axis from -5 to 105, y-axis from 0 to 5

plt.xticks([10 * i for i in range(11)]) # x-axis labels at 0, 10, ... 100
plt.xlabel("Decile")
plt.ylabel("# of students")
plt.title("Distribution of Exam 1 Grades")
#plt.show()
plt.savefig('im/scores_numbers.png')
plt.close()

# Huge difference bar chart
mentions = [500, 506]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard somone says 'data science'")

plt.ticklabel_format(useOffset = False)

# misleading y-axis only shows the part above 500
plt.axis([2016.5, 2018.5, 499, 506])

# Not so huge increase
# plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Look at the Huge Increase")
#plt.show()
plt.savefig('im/huge_increase.png')
plt.close()

# lines chart
variance = [2 ** x for x in range(9)]
bias_squared = variance[::-1]   # reverse
#print("variance, bias_squared = ", variance, bias_squared)

total_error = [x+y for x, y in zip(variance, bias_squared)]

xs = [i for i, _ in enumerate(variance)]

# make multiple calls to plot.plot
# to show multiple series on the same chart
plt.plot(xs, variance, 'g-', label = 'variance') #green solid line
plt.plot(xs, bias_squared, 'r-.', label = 'bias^2') # red dot-dashed line
plt.plot(xs, total_error, 'b:', label = 'total_error') # blue dotted line

# a legend for free (loc = 9 means 'top center')
plt.legend(loc = 9)
plt.xlabel("model complxity")
plt.xticks([])
plt.title('The Bias-Variance Tradeoff')
#plt.show()
plt.savefig('im/variance_bias^2.png')
plt.close()

# scatter plot
friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# label each points
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy = (friend_count, minute_count), # put the label with its points
                 xytext = (5,-5),    # slightly offset ???
                 textcoords = 'offset points')

plt.title("Daily minutes vs. # of Friends")
plt.xlabel("# of Friends")
plt.ylabel("daily minutes spent on the site")
# plt.show()
plt.savefig('im/friends_minutes.png')
plt.close()
