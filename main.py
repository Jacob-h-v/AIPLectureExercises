
import numpy as np
import matplotlib.pyplot as plt

# Check whether plots are interactive or not ... interactive mode is most useful from the command line
if plt.isinteractive() == True:
    print("Interactive mode is TRUE - exiting.")
    exit()
else:
    print("interactive mode is FALSE - remember to close graph window.")


# Generate 1000 samples from a normal distribution
mean = 4.0
std_dev = 1.0
n_samples = 1000
samples = np.random.normal(mean, std_dev, n_samples)

# Computing Probability Density Function of Normal Distribution
def normal_pdf(pdfx, pdfmean, standard_dev):
    # Params:
    # x: float or numpy.ndarray: input values
    # mean: float: the mean or average of the distribution
    # standard_dev: float: the standard deviation (a measure of dispersion) of the distribution
    # returns: float or numpy.ndarray: The probability density function evaluated at the given point(s).
    return (1/(standard_dev * np.sqrt(2*np.pi))) * np.exp(-((pdfx - mean) ** 2) / (2 * standard_dev ** 2))

# Computing Z-score (standard score)
def z_score(zsample, zmean, ztandard_dev):
    # Params:
    # zsample: float or numpy.ndarray: input values (x value)
    # mean: float: the mean or average of the distribution
    # ztandard_dev: float: the standard deviation (a measure of dispersion) of the distribution
    return ((zsample - zmean) / ztandard_dev)

# Running the normal_pdf definition
pdf_values = normal_pdf(samples, mean, std_dev)
for sample, pdf in zip(samples[:5], pdf_values[:5]):
    print(f"Sample = {sample:.2f}, PDF = {pdf:.5f}")
# print pdf of specific sample value:
specificsample = 5
print(f"PDF of given specific sample value ({specificsample}): {normal_pdf(specificsample, mean, std_dev)}")

# Running the z_score definition (standard score)
zsample = 4
print(f"Z-score of specific sample value: {zsample} is {z_score(zsample, mean, std_dev)}")

# Plot histogram
plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Histogram of 1000 Samples from Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')

#plt.show will halt thread - program cannot proceed until open windows have been closed
plt.show()

#compute the mean of the samples in over the first 10, the first 20, the first 30,... samples
step_count = 0
step_size = 10
mean_values = np.zeros(np.uint(n_samples/step_size))
for end_index in range(step_size, n_samples + 1, step_size):
    mean_values[step_count] = np.mean(samples[0:end_index])
    step_count = step_count + 1

plt.plot(mean_values)
plt.xlabel("Number of samples included in mean, times {}".format(step_size))
plt.ylabel('Mean over samples')
plt.show()

exit()