There was an issue with the idea behind the first & second data sets. The first part of the (noisy) curve is not very well correlated to the middle part of the curve because there are some additional interactions inside the detector; a lot of variance in the shape. Henceforth, not a lot of predictive power by using these covariates.

What Myroslav did instead was to look only at the beginning part of the curves, which should be most relevant. How much constitutes “the beginning” is a hyperparameter that we may teak later; as of now, consider only the part of the curve up until the $20%$ mark. 

To make the curve smoother, he tried various functions. The one that gave good results was a 3rd degree polynomial fit. It is imperfect, because before the pulse shape the curve should be flat, and then start rising with the pulse. In reality, there is an ever so slight dip in the curve before that, but we ignore this issue because otherwise, the fit is super nice!

To find the root of this pulse shape, he tried both the CFD algorithm (which overall gave worse results than before) and Newton’s algorithm, to find the root. In the image below, that is T_1 or T_{zc} (zero-crossing). It is a number in reference to a fixed origin $t$, which does not have a great physical importance. 

The “true” time, from the reference detector, is $T_0$ on the image below, or T_{zero}. It is given in reference to the same origin $t$ from before, so the difference $dt = T_{zero} - T_{zc}$ should be origin independent. Dt is the number that we must accurately predict, using ML. The most promising part of the new data set: the dt is correlated to the coefficients of this polynomial fit and to the chi^2 value of the fit. 



Important to note, the way we got $T_{zc}$ in Data 1 was through CFD, which was less accurate than the method based on Newton. So the dt that we had to predict was more garbage, and also with covariates that didn’t relate to it strongly. Now the method that Myroslav used to preprocess the data gives a new root for the HPGe pulse shape that is close to the true root within a dt correction (dt- objective to dynamically correct using Machine Learning). The distribution of corrections, due to the improvement, is already much narrower than the dt corrections from before, using CFD; it increased the resolution from 15 nm to about 7 nm. I have to do even better than that!

The difference $T_{zero} - T_{zc}$ is, ideally, a constant. In reality it is a distribution that kinda looks like a Gaussian, but with a tail towards high values. Alternative objective: get rid of the tail (to make it even thinner). 

In other words: $T_{zc}$ is something that we can get from the pulse shape in live acquisition. 
$T_{zc}$ through raw CFD gives an interval of width 15 nm that we are reasonably sure will cover the true time of hitting the detector
$T_{zc} + dt$, where $dt$ is chosen smartly through ML (chosen based on the pulse shapes) will move this number closer to the true value, and also allow us to shrink the width of this interval; the smarter the correction, the thinner the interval
The distribution of dt’s below already suggests that if we add the mean (\approx 130 ns), then we are close to the true value within 7 ns.
But using also information from the pulse shapes, we hope to do even better.

In the figure below, the order of the subtraction may be reversed. See which is the bigger number.


Who are the covariates?

Ch: Channel where the data comes from. To be ignored
E: Energy deposition inside the detector. While physically irrelevant, it is good to select only the features that have an energy of the 511 keV peak (4490 - 4570) - helps with the more precise timing of T_{zero}
T_{zero}, T_{zc}: See above the meaning of each
dT: (?) The difference between T_{zero} and T_{zc}
a0, a1, a2, a3: The coefficients of the polynomial fit \sum_i^3 ai * x^i
Chi2 = The chi^2 value of the (?) polynomial fit to the pulse shape. It is divided by the degrees of freedom, but not scaled by the variance.

Objectives:
Predict $T_ML$ such that $T_ML + T_{zc} = T_{zero} + C$, with C = 0 (“vanilla”) or C = mean(-T_{zero} - + T_{zc}) \approx 130 - i.e. add smart correction on top of the average of 130 ns. OR
Somehow find a correlation between the T_{zero} - T_{zc} values that are past the peak (in the tail) and other covariates. Maybe use ML to find these, and get rid of them systematically. Classifiers?

Note: I can use “scatter_matrix” from Pandas (https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html) to easily have the histogram and pairwise scatter plots of all features; much easier than the detailed plots from my notebook

Note: To apply Boolean masking on a Pandas data frame, try e.g.
>> df.loc( (data[“E”] > 4490) & (data[“E”] < 4570) )
