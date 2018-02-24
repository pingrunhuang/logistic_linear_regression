# Cost Function
Cost function is the measurement that I are going to evaluate our linear regression model. Suppose our target model is h(x) = theta(0) + theta(1)x, the cost function will look like the following:
J(theta(0), theta(1)) = 1/2 * 1/m * sum(square(h(x(i)) - y(i)))  
which is also called Squared Error Function or Mean Squared Error(MSE)
#### But why do we use this cost function instead of others?
Mathematics have already proven that this function is a convex function which guarantee us to always get the global optimal.
Our goal is to get the appropriate theta(0) and theta(1) so the the value of J could be the smallest.



# Gradient descent

* should simultaneous update theta(0) and theta(1)
temp_theta0 = theta0 - alpha * np.diff(J(theta0, theta1), theta0) = theta0 - alpha * 1/m * sum(h(x(i)) - y(i))
temp_theta1 = theta1 - alpha * np.diff(J(theta0, theta1), theta1) = theta1 - alpha * 1/m * sum((h(x(i)) - y(i)) * x(i))
theta0=temp_theta0
theta1=temp_theta1

* turns out that the gradient decent will always get the global optimal.

* different kind of gradient descent:
  * Batch gradient descent:


# Feature scaling
* Feature scaling is divide the a column with the largest value of the inside that column (v / max(column)), this will shorten the time to find the global optimization
* feature should not too large or too small, ideally scale to 2
* normalizing the value with (x[i] - mean(x)) / (max(x) - min(x)) before hand (recommended by Andrew)
* plot cost function in action.
* alpha too small: might converge very slow; alpha too large: might not converge

# Normal equation
Further more, theta = inverse(X' * X) * X' * Y will let us get the optimal J(theta). In this case, you do not need to normalize the data and you get the final result in one step. However, if you have too many columns(features), you end up with large amount of compute resource.


# validation
use root mean squared error (RMSE) to get the validation.
RMSE = sqrt(sum(square(observed - predict) / len(observed)))
