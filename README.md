# CMEvelocity
 
A novel method for analysis of CME kinematics using Computer vision (opencv2) in Python to recognise and calculate the distance of the front of the CME from the solar centre. This can then be used to find the velocity over time. Steps for this analysis are as follows:
- Convert to Grayscale
- 2 x Median Blur, n = 5
- 3 x Gaussian Blur, n = 5
- Thresholding (values between 200-255)
- Contour Finding
- Distance between furthest point in contour and solar centre

## Example

Figure ? Shows the results of this method, which gave an average velocity of 1222 m/s. This is similar to the velocity given by the SOHO/LASCO catalog.
