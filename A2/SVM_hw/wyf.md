# k(x, x_i) = (x^Tx_i)^3DDA2020 Assignment2

119010317 Yifan WANG



## Written Question

### 1.

a.

<img src="E:\university\32\DDA2020\assignment\assignment2\1.jpg" alt="1" style="zoom:50%;" />
$$
p(y=1|x, w) = \frac{1}{1+exp(-w_0-w_1x_1-w_2x_2)}\\
L(w) = \prod_i^np(y=1|x_i, w) = \prod_i^n(1+exp(-wx)^{-1})^{y_i}(1-(1+exp(-wx)^{-1})^{n-y_i}\\
$$
this problem can be solved with gradient descent way, due to the different iteration number, initial state and learning rate we choose, the result may be different.

the error rate is: 0%

b.

<img src="E:\university\32\DDA2020\assignment\assignment2\2.jpg" alt="2" style="zoom:50%;" />

there's one data point that be misclassified, the error rate is: 1/13 = 0.08

c.

<img src="E:\university\32\DDA2020\assignment\assignment2\3.jpg" alt="3" style="zoom:50%;" />

there's two data points that be misclassified, the error rate is: 2/13 = 0.15

d.

<img src="E:\university\32\DDA2020\assignment\assignment2\4.jpg" alt="4" style="zoom:50%;" />

the error rate is: 0%

### 2.

$$
\phi(x_1) = [1, 0, 0]^T\\
\phi(x_2) = [1, 2, 2]^T
$$

a.

because the vector \phi(x_1)-\phi(x_2) must be  perpendicular to the dicision boundary, so it's parallel to w
$$
v = [0, 2, 2]^T
$$
b. 

because there's only 2 points, so the length of margin half of the length of vector \phi(x_1)-\phi(x_2)
$$
\phi(x_1)\phi(x_2) = 2\sqrt 2\\
margin = \sqrt2
$$
c.
$$
margin = \sqrt 2 = \frac{1}{||\bold w||}, \ \ ||\bold w|| = \frac{\sqrt 2}{2}\\
$$
so we can get:
$$
\bold w = [0, \frac{1}{2}, \frac{1}{2}]^T
$$
d.
$$
y_1(\bold w^T\phi(x)+w_0) = 1\\
y_2(\bold w^T\phi(x)+w_0) = 1\\
$$
take in all data:
$$
-w_0 = 1\\
2 + w_0 = 1\\
$$
so, the final answer is:
$$
w_0 = -1
$$
e.
$$
f(x) = -1 + \frac{\sqrt 2}{2}x + \frac{1}{2}x^2
$$

### 3.

$$
min \ \ \frac{1}{2}||w||^2 + C\sum \xi_i
$$

for the KKT condition, we have:
$$
L(w, b, \alpha) = \frac{1}{2}||w||^2 + C\sum \xi_i + \sum[\alpha_i(1-\xi_i-y_i(w^Tx_i + b) - \mu_i\xi_i]\\ 
$$
the KKT condition can't gurantee \xi to be smaller than 1. If we set a very small C, it may happens that, we can increase ||w|| a lot than the case we set a larger C, but this new w will cause misclassification. 

### 4.

（1）.


$$
min _{b_0, W} \ \frac{1}{2}||\omega||^2\\
s.t. \ 	y_i(b+w ·x_i) \geq 1
$$
dual problem:
$$
max_{\alpha} \sum_i \alpha - \frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
s.t. \ \ \sum_i^m\alpha_iy_i = 0, \ \ \alpha_i \geq0
$$
the final solution is:
$$
w = [-1, -1]^T \\
b=0
$$
(2).

support vectors:
$$
(1, 0), (0, 1), (-1, 0), (0, -1)
$$

(3).

-1-2=-3<0

Class -1

### 5.

(1).

Yes, we can use kernel to reflect those data point to higher dimension, then it can be seperate

(2).

Class -1:
$$
\phi(x_1) = (1, 0)\\
\phi(x_2) = (0, 1)\\
\phi(x_3) = (1, 0)\\
\phi(x_4) = (0, 1)
$$
Class 1:
$$
\phi(x_1) = (4, 0)\\
\phi(x_2) = (0, 4)\\
\phi(x_3) = (4, 0)\\
\phi(x_4) = (0, 4)
$$



$$
min _{b_0, W} \ \frac{1}{2}||\omega||^2\\
s.t. \ 	y_i(b+w ·x_i) \geq 1
$$
dual problem:
$$
max_{\alpha} \sum_i \alpha - \frac{1}{2}\sum_{ij}\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
s.t. \ \ \sum_i^m\alpha_iy_i = 0, \ \ \alpha_i \geq0
$$
the final solution is:
$$
w = [\frac{2}{3}, \frac{2}{3}]^T \\
b=-\frac{5}{3}
$$

(3). 
$$
f(1, 2) = \frac{2}{3}(1+4)-\frac{5}{3}  = \frac{5}{3} > 0
$$
so it belongs to +1

### 6.

we know that:
$$
\gamma = \frac{|w^Tx+b|}{||w||}
$$
from the assumption of maximum-margin classifier, for those support vectors, we have:
$$
|w^Tx+b| = 1
$$
so, we have:
$$
\gamma = \frac{1}{||w||}\\
\frac{1}{\gamma^2} = ||w||^2
$$
Now, conduct the dual problem:

original problem:
$$
min \ \ \ \frac{1}{2}||w||^2 \\
s.t. \ \ \  1 - y_i(w^Tx_i + b) \leq 0
$$
Lagrange  function:
$$
L(w, b, a) = \frac{1}{2}||w||^2 + \sum_i\alpha_i(1-y_i(w^Tx_i+b))
$$
KKT condition:

​		Stationary:
$$
\frac{\partial L}{\partial w} = 0, \ w = \sum \alpha_iy_ix_i\\
\frac{\partial L}{\partial b} = 0, \ \ \sum \alpha_iy_i = 0\\
$$
​		primal:
$$
\alpha_i \geq 0, \ \ 1 - y_i(x^Tx_i + b) \leq 0
$$
​		complementary:
$$
\alpha_i(1-y_i(w^Tx_i+b)) = 0
$$
according to kkt condition, we further calculate ||w||^2:
$$
||w||^2 = w^T\sum \alpha_i y_ix_i =\sum \alpha_iy_i(w^Tx_i)
$$
because:
$$
\sum \alpha_i y_i = 0
$$
so, if we add one more constant term, the result won't change, which is:
$$
\sum b\alpha_iy_i = 0\\
||w||^2 = w^T\sum \alpha_i y_ix_i =\sum \alpha_iy_i(w^Tx_i) = \sum \alpha_iy_i(w^Tx_i+b)
$$
for maximum-margin classifier, we have:
$$
y_i(w^Tx_i+b) = 1
$$
so it's easy to know that:
$$
\alpha_iy_i(w^Tx_i+b) = \alpha_i, \ \ \ \ \ \alpha_i \neq 0 \\
\alpha_iy_i(w^Tx_i+b) = 0, \ \ \ \ \ \ \alpha_i = 0
$$
so,
$$
||w||^2 = w^T\sum \alpha_i y_ix_i =\sum \alpha_iy_i(w^Tx_i) = \sum \alpha_iy_i(w^Tx_i+b) = \sum\alpha_i\\
\frac{1}{\gamma^2} =  \sum\alpha_i
$$


## Programming Question

### Introduction

#### Date set

This problem use iris data set. There's 3 class, setosa, versicolor and virginica. Each class has 50 samples in training set and 10 samples in test set.

#### Methods

This report make a muti-class classification with one vs rest strategy.  Three svm models are built with `sklearn.svm.SVC`, in each model, only one class are set to be 1, the other 2 classes will be set to -1. the point will be classified into the class which has the largest probability.

The random set is set to be 42.

#### Models

In this report, 6 different svm models are implemented. They're linear svm, svm with slack variable, svm with polynomial kernel n=2 and n=3, svm with rbf kernel and svm with sigmoid kernel

##### linear svm 

the model form is:
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2\\
s.t. 1-y_i(\bold w^T \bold x_i + b) \leq 0, \forall i
$$
in our result, we calculate training error, w, b, the index of support vectors

##### linear svm with slack variable

the model form is
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2 + C\sum_i \xi_i\\
s.t. 1-y_i(\bold w^T \bold x_i + b) \leq 0, \forall i
$$
Lagrange function:
$$
L(\bold w, b, \alpha) = \frac{1}{2}||\bold w||^2 + C\sum_i^m\xi_i + \sum_i^m[\alpha_i(1-\xi_i-y_i(\bold w^Tx_i+b)) - u_i\xi_i]
$$
KKT condition:

stationarity:
$$
\frac{\partial L}{\partial \bold w} = 0,  \ \  \bold w = \sum_i^m\alpha_iy_ix_i\\
\frac{\partial L}{\partial b} = 0, \ \  \sum_i^ma_iy_i=0\\
\frac{\partial L}{\partial \xi_i} = 0, \ \  \alpha_i=C-\mu_i, \forall i
$$
feasibility:
$$
\alpha_i \geq 0, 1-\xi_i-y_i(\bold w^Tx_i+b) \leq 0, \xi_i \geq 0, \mu_i \geq 0, \forall i
$$
complementray slackness:
$$
\alpha_i(1-\xi_i-y_i(\bold w^Tx_i+b)) = 0, \mu_i\xi_i=0, \forall i
$$
so, we can derive the dual problem:
$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_iy_iy_ix_i^Tx_j\\
s.t.\sum_i^m\alpha_iy_i=0, 0 \leq \alpha_i \leq C, \forall i
$$
Here, C is set to 0.1, 0.2……1.0 respectively.

in our result, we calculate training error, w, b, the index of support vectors

##### 2nd-order polynomial kernel

the model form is
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2 + C\sum_i \xi_i\\
s.t. 1-y_i(\bold w^T \phi(x_i) + b) \leq 0, \forall i
$$
Lagrange function:
$$
L(\bold w, b, \alpha) = \frac{1}{2}||\bold w||^2 + C\sum_i^m\xi_i + \sum_i^m[\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) - u_i\xi_i]
$$
KKT condition:

stationarity:
$$
\frac{\partial L}{\partial \bold w} = 0,  \ \  \bold w = \sum_i^m\alpha_iy_i\phi(x_i)\\
\frac{\partial L}{\partial b} = 0, \ \  \sum_i^ma_iy_i=0\\
\frac{\partial L}{\partial \xi_i} = 0, \ \  \alpha_i=C-\mu_i, \forall i
$$
feasibility:
$$
\alpha_i \geq 0, 1-\xi_i-y_i(\bold w^T\phi(x_i)+b) \leq 0, \xi_i \geq 0, \mu_i \geq 0, \forall i
$$
complementray slackness:
$$
\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) = 0, \mu_i\xi_i=0, \forall i
$$
so, we can derive the dual problem:
$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_iy_iy_i\phi(x_i)^T\phi(x_j)\\
$$
use the kernel:
$$
k(x, x_i) = (\gamma(ax^Tx_i + b)^N
$$
take a = 1, b = 0, gamma = 1, N=2
$$
k(x, x_i) = (x^Tx_i)^2
$$

$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_jy_iy_j(x^Tx_i)^2\\
s.t.\sum_i^m\alpha_iy_i=0, 0 \leq \alpha_i \leq C, \forall i
$$
in our result, we calculate training error,  b, the index of support vectors

##### 3nd-order polynomial kernel

the model form is
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2 + C\sum_i \xi_i\\
s.t. 1-y_i(\bold w^T \phi(x_i) + b) \leq 0, \forall i
$$
Lagrange function:
$$
L(\bold w, b, \alpha) = \frac{1}{2}||\bold w||^2 + C\sum_i^m\xi_i + \sum_i^m[\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) - u_i\xi_i]
$$
KKT condition:

stationarity:
$$
\frac{\partial L}{\partial \bold w} = 0,  \ \  \bold w = \sum_i^m\alpha_iy_i\phi(x_i)\\
\frac{\partial L}{\partial b} = 0, \ \  \sum_i^ma_iy_i=0\\
\frac{\partial L}{\partial \xi_i} = 0, \ \  \alpha_i=C-\mu_i, \forall i
$$
feasibility:
$$
\alpha_i \geq 0, 1-\xi_i-y_i(\bold w^T\phi(x_i)+b) \leq 0, \xi_i \geq 0, \mu_i \geq 0, \forall i
$$
complementray slackness:
$$
\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) = 0, \mu_i\xi_i=0, \forall i
$$
so, we can derive the dual problem:
$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_iy_iy_i\phi(x_i)^T\phi(x_j)\\
$$
use the kernel:
$$
k(x, x_i) = (\gamma(ax^Tx_i + b)^N
$$
take a = 1, b = 0, gamma = 1, N=3
$$
k(x, x_i) = (x^Tx_i)^3
$$

$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_jy_iy_j(x^Tx_i)^3\\
s.t.\sum_i^m\alpha_iy_i=0, 0 \leq \alpha_i \leq C, \forall i
$$
in our result, we calculate training error,  b, the index of support vectors

##### rbf kernel

the model form is
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2 + C\sum_i \xi_i\\
s.t. 1-y_i(\bold w^T \phi(x_i) + b) \leq 0, \forall i
$$
Lagrange function:
$$
L(\bold w, b, \alpha) = \frac{1}{2}||\bold w||^2 + C\sum_i^m\xi_i + \sum_i^m[\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) - u_i\xi_i]
$$
KKT condition:

stationarity:
$$
\frac{\partial L}{\partial \bold w} = 0,  \ \  \bold w = \sum_i^m\alpha_iy_i\phi(x_i)\\
\frac{\partial L}{\partial b} = 0, \ \  \sum_i^ma_iy_i=0\\
\frac{\partial L}{\partial \xi_i} = 0, \ \  \alpha_i=C-\mu_i, \forall i
$$
feasibility:
$$
\alpha_i \geq 0, 1-\xi_i-y_i(\bold w^T\phi(x_i)+b) \leq 0, \xi_i \geq 0, \mu_i \geq 0, \forall i
$$
complementray slackness:
$$
\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) = 0, \mu_i\xi_i=0, \forall i
$$
so, we can derive the dual problem:
$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_iy_iy_i\phi(x_i)^T\phi(x_j)\\
$$
use the kernel:
$$
k(x, x_i) = exp(-\gamma||x-x_i||^2), \gamma > 0
$$
take gamma = 1/2
$$
k(x, x_i) = exp(-\frac{||x-x_i||^2}{2})
$$

$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_jy_iy_jexp(-\frac{||x-x_i||^2}{2})\\
s.t.\sum_i^m\alpha_iy_i=0, 0 \leq \alpha_i \leq C, \forall i
$$
in our result, we calculate training error,  b, the index of support vectors

##### Sigmoid kernel

the model form is
$$
min_{\bold w, b} \frac{1}{2}||\bold w||^2 + C\sum_i \xi_i\\
s.t. 1-y_i(\bold w^T \phi(x_i) + b) \leq 0, \forall i
$$
Lagrange function:
$$
L(\bold w, b, \alpha) = \frac{1}{2}||\bold w||^2 + C\sum_i^m\xi_i + \sum_i^m[\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) - u_i\xi_i]
$$
KKT condition:

stationarity:
$$
\frac{\partial L}{\partial \bold w} = 0,  \ \  \bold w = \sum_i^m\alpha_iy_i\phi(x_i)\\
\frac{\partial L}{\partial b} = 0, \ \  \sum_i^ma_iy_i=0\\
\frac{\partial L}{\partial \xi_i} = 0, \ \  \alpha_i=C-\mu_i, \forall i
$$
feasibility:
$$
\alpha_i \geq 0, 1-\xi_i-y_i(\bold w^T\phi(x_i)+b) \leq 0, \xi_i \geq 0, \mu_i \geq 0, \forall i
$$
complementray slackness:
$$
\alpha_i(1-\xi_i-y_i(\bold w^T\phi(x_i)+b)) = 0, \mu_i\xi_i=0, \forall i
$$
so, we can derive the dual problem:
$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_iy_iy_i\phi(x_i)^T\phi(x_j)\\
$$
use the kernel:
$$
k(x, x_i) = \tanh(\gamma x^Tx + r)
$$
take gamma = 1/4 because x is 4 dimensional, r = 0
$$
k(x, x_i) = \tanh(\frac{1}{4} x^Tx)
$$

$$
max_{\alpha_i} \sum_i^m\alpha_i-\frac{1}{2}\sum_{i, j}\alpha_i\alpha_jy_iy_j\tanh(\frac{1}{4} x^Tx)\\
s.t.\sum_i^m\alpha_iy_i=0, 0 \leq \alpha_i \leq C, \forall i
$$
in our result, we calculate training error,  b, the index of support vectors
